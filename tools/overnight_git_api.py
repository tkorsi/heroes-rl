"""Exact Git-object transport through the existing authenticated GitHub API proxy.

No credential is read, created or printed. urllib retains the managed HTTPS
proxy and CA trust. Branches only advance by fast-forward, after every blob,
tree and original commit SHA matches the local immutable Git object.
"""
import argparse
import base64
from datetime import datetime,timedelta,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

GIT='/usr/local/bin/git'
BRANCH='codex/teacher-overnight-20261026'
ROOTS={Path('/workspace/heroes-overnight'):'tkorsi/heroes-rl',
       Path('/workspace/heroes-overnight-private'):'tkorsi/heroes-rl-training-state'}
BASE=Path('/workspace/Heroes-artifacts/overnight-20261026')
COMPUTE=1793008500.;HARD=1793008800.


def require(condition,message):
    if not condition:raise ValueError(message)


def deadline():
    return COMPUTE if os.environ.get('HEROES_OVERNIGHT_ROLE')=='compute' else HARD-15


def git(root,*args,input=None):
    return subprocess.check_output([GIT,'-C',str(root),*args],input=input,timeout=45)


def api(repo,path,method='GET',value=None):
    data=json.dumps(value,ensure_ascii=False).encode() if value is not None else None
    request=urllib.request.Request('https://api.github.com/repos/'+repo+('/'+path if path else ''),data=data,method=method,
        headers={'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'heroes-teacher-exact-git-transport'})
    for attempt in range(3):
        left=deadline()-time.time();require(left>2,'Absolute Git API deadline')
        try:
            with urllib.request.urlopen(request,timeout=min(45,left-1)) as response:return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429,502,503,504) or attempt==2:raise RuntimeError('GitHub API HTTP '+str(error.code)) from None
            time.sleep(min(3,max(.05,deadline()-time.time()-1)))


def identity(line):
    match=re.fullmatch(r'(.+) <([^<>]*)> (\d+) ([+-])(\d{2})(\d{2})',line)
    require(match is not None,'Unsupported Git identity encoding')
    name,email,seconds,sign,hours,minutes=match.groups()
    offset=(int(hours)*60+int(minutes))*(1 if sign=='+' else -1)
    date=datetime.fromtimestamp(int(seconds),timezone(timedelta(minutes=offset))).isoformat()
    return dict(name=name,email=email,date=date)


def commit_body(raw):
    header,message=raw.decode('utf-8').split('\n\n',1);values={};parents=[]
    for line in header.splitlines():
        key,value=line.split(' ',1)
        require(key in ('tree','parent','author','committer'),'Signed/extended commits require separate exact handling')
        if key=='parent':parents.append(value)
        else:require(key not in values,'Duplicate Git header');values[key]=value
    return dict(tree=values['tree'],parents=parents,author=identity(values['author']),
        committer=identity(values['committer']),message=message)


def remote_head(repo):
    result=api(repo,'git/ref/heads/'+BRANCH)
    sha=result['object']['sha'];require(re.fullmatch('[0-9a-f]{40}',sha) is not None,'Invalid remote ref')
    return sha


def changes(root,parent,commit):
    rows=[]
    raw=git(root,'diff-tree','-r','--no-commit-id','--raw','--no-renames','-z',parent,commit).split(b'\0')
    for index in range(0,len(raw)-1,2):
        meta=raw[index].decode().split();path=raw[index+1].decode()
        require(not Path(path).is_absolute() and '..' not in Path(path).parts,'Unsafe tree path')
        rows.append(dict(path=path,mode=meta[1],sha=meta[3],deleted=meta[4]=='D'))
    return rows


def push(root):
    root=Path(root).resolve();require(root in ROOTS,'Only the two authorized checkouts may publish')
    repo=ROOTS[root];lock=(BASE/('git-api-'+repo.split('/')[-1]+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    info=api(repo,'');require(info['full_name']==repo,'Repository identity differs')
    if root.name.endswith('-private'):require(info['private'] is True,'Raw state repository must remain private')
    branch=git(root,'branch','--show-current').decode().strip();require(branch==BRANCH,'Unexpected publication branch')
    require(git(root,'remote','get-url','origin').decode().strip()=='https://github.com/'+repo+'.git','Unexpected origin')
    head=git(root,'rev-parse','HEAD').decode().strip();remote=remote_head(repo)
    if head==remote:return dict(commit=head,remote_acknowledged=True,transport='existing-github-api')
    subprocess.run([GIT,'-C',str(root),'merge-base','--is-ancestor',remote,head],check=True,timeout=30)
    commits=git(root,'rev-list','--reverse','--topo-order',remote+'..'+head).decode().splitlines()
    count=0
    for commit in commits:
        body=commit_body(git(root,'cat-file','commit',commit));require(body['parents'],'Unexpected root commit')
        parent=body['parents'][0];base_tree=git(root,'rev-parse',parent+'^{tree}').decode().strip();elements=[]
        for row in changes(root,parent,commit):
            if repo=='tkorsi/heroes-rl':
                require(not row['path'].lower().endswith(('.pt','.pth','.lod','.vid','.snd','.zip','.tar.gz')),'Private binary cannot enter public source')
            if row['deleted']:elements.append(dict(path=row['path'],mode='100644',type='blob',sha=None));continue
            require(row['mode'] in ('100644','100755','120000'),'Only ordinary/symlink blobs are supported')
            payload=git(root,'cat-file','blob',row['sha'])
            created=api(repo,'git/blobs','POST',dict(content=base64.b64encode(payload).decode(),encoding='base64'))
            require(created['sha']==row['sha'],'Remote blob differs from exact local bytes')
            elements.append(dict(path=row['path'],mode=row['mode'],type='blob',sha=row['sha']));count+=1
        tree=api(repo,'git/trees','POST',dict(base_tree=base_tree,tree=elements))
        require(tree['sha']==body['tree'],'Remote Git tree differs')
        created=api(repo,'git/commits','POST',body)
        require(created['sha']==commit,'Remote original Git commit differs; never retarget metadata')
    require(remote_head(repo)==remote,'Remote ref moved during publication; stop instead of forcing')
    updated=api(repo,'git/refs/heads/'+BRANCH,'PATCH',dict(sha=head,force=False))
    require(updated['object']['sha']==head and remote_head(repo)==head,'Remote commit acknowledgement differs')
    result=dict(commit=head,remote_acknowledged=True,transport='existing-github-api',commits=len(commits),blobs=count,
        original_commit_metadata_preserved=True,tree=git(root,'rev-parse',head+'^{tree}').decode().strip())
    with (BASE/'git-api-publications.jsonl').open('a') as stream:stream.write(json.dumps(dict(repository=repo,**result))+'\n')
    return result


def shim(arguments):
    args=list(arguments);root=Path.cwd()
    if len(args)>=2 and args[0]=='-C':root=Path(args[1]).resolve();args=args[2:]
    if root in ROOTS and args==['push','origin','HEAD:refs/heads/'+BRANCH]:
        print(json.dumps(push(root)));return 0
    if root in ROOTS and args==['ls-remote','origin','refs/heads/'+BRANCH]:
        print(remote_head(ROOTS[root])+'\trefs/heads/'+BRANCH);return 0
    os.execv(GIT,[GIT,*arguments])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--push',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(push(args.push)))
