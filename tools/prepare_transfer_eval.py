"""Prepare immutable Git checkpoint inputs and prospectively varied authored maps."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/Heroes-artifacts/transfer-eval-20261026')
PRIVATE=Path('/workspace/heroes-rl-training-state')
NIGHT=Path('/workspace/Heroes-artifacts/overnight-20261026')
PAIRS=[((4,2),(6,3)),((4,2),(7,7)),((4,2),(2,4)),((4,2),(0,7)),
       ((7,2),(1,1)),((7,2),(2,3)),((7,2),(7,4)),((7,2),(0,7)),
       ((4,6),(6,6)),((4,6),(7,0)),((4,6),(0,1)),((4,6),(2,3)),
       ((7,6),(1,6)),((7,6),(0,0)),((7,6),(6,3)),((7,6),(2,7))]


def binding(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def git_file(commit,path):
    return subprocess.check_output(['git','-C',str(PRIVATE),'show',commit+':'+path],timeout=30)


def maps():
    result={}
    for seat in (0,1):
        original=NIGHT/f'maps/curriculum-3-{seat}.vmap'
        with zipfile.ZipFile(original) as z:base={n:json.loads(z.read(n)) for n in z.namelist()}
        for index,((tx,ty),(hx,hy)) in enumerate(PAIRS):
            value=json.loads(json.dumps(base));objects=value['objects.json']
            hero=next(o for o in objects if o['type']=='hero');town=next(o for o in objects if o['type']=='town')
            assert len(objects)==2 and hero['options']['army']==[{'type':'core:pikeman','amount':100}]
            hero.update(x=hx,y=hy);town.update(x=tx,y=ty)
            value['header.json']['players'][('red','blue')[1-seat]]['mainTown'].update(x=tx-2,y=ty)
            value['header.json']['name']=f'Frozen transfer easy8x8 layout{index} seat{seat}'
            assert all(isinstance(tile,str) and tile.startswith('dt') for row in value['surface_terrain.json'] for tile in row),'Only native passable dirt terrain is permitted'
            # Conservative entire five-by-three town footprint, except its visitable cell.
            blocked={(x,y) for x in range(tx-4,tx+1) for y in range(ty-2,ty+1)}
            target=(tx-2,ty);blocked.discard(target)
            assert (hx,hy) not in blocked and (hx,hy)!=target
            reached={(hx,hy)};pending=[(hx,hy)]
            while pending:
                x,y=pending.pop()
                for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                    nxt=(x+dx,y+dy)
                    if 0<=nxt[0]<8 and 0<=nxt[1]<8 and nxt not in blocked and nxt not in reached:
                        reached.add(nxt);pending.append(nxt)
            assert target in reached,'Authored start has no conservative open route'
            assert value['surface_terrain.json']==base['surface_terrain.json']
            path=BASE/f'maps/easy-layout{index:02d}-seat{seat}.vmap'
            assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
                for name,data in value.items():
                    info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
                    z.writestr(info,json.dumps(data,sort_keys=True,indent=2)+'\n')
            result[f'easy-{index}-{seat}']=dict(**binding(path),layout=index,seat=seat,hero=[hx,hy],town_anchor=[tx,ty],
                town_visitable=list(target),chebyshev_distance=max(abs(hx-target[0]),abs(hy-target[1])),
                conservative_collision_and_connectivity_checked=True,terrain_unchanged=True,
                validation_not_supplied_to_policy=True)
    original=ROOT/'experiments/explorer/maps/hero-management-v2.vmap'
    assert binding(original)['sha256']=='ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3'
    dest=BASE/'maps/hero-management-v2.vmap';shutil.copyfile(original,dest);result['original']=binding(dest)
    for seat in (0,1):result[f'gate-{seat}']=binding(NIGHT/f'maps/curriculum-3-{seat}.vmap')
    return result


def checkpoints():
    historical='bcdee6ecc3609f3024f304c15ef8ccba7e605404'
    records={146:(historical,'checkpoints/local-v8-update92/v8-update92-repair3-continuation-148'),
        254:('2f8d587c7a4a82f6b08bb752a1322ea31fd100df','checkpoints/overnight-20261026/imported-final-local'),
        412:('5577155c0b9e9d859d795edc0831d9d8e800d9d6','checkpoints/overnight-20261026/campaign-v1/http-transport-003/train-0265-seat1')}
    result={}
    for update,(commit,prefix) in records.items():
        folder=BASE/f'inputs/update{update}';folder.mkdir(parents=True,exist_ok=False)
        files={}
        for name in ('plan.json','report.json','after-policy.pt','after-resume.pt'):
            raw=git_file(commit,prefix+'/'+name);p=folder/name;p.write_bytes(raw);files[name]=binding(p)
        plan=json.loads((folder/'plan.json').read_text());report=json.loads((folder/'report.json').read_text())
        assert report['plan_sha256']==files['plan.json']['sha256'] and report['cumulative_progress']['updates']==update
        for kind in ('policy','resume'):assert report['after_checkpoint'][kind]['sha256']==files[f'after-{kind}.pt']['sha256']
        dependencies={}
        for kind,bound in [('checkpoint',plan['checkpoint']),('metadata_initial',plan['checkpoint_metadata']['initial_checkpoint'])]:
            digest=bound['sha256'];candidates=[]
            if update==146:candidates=[git_file(historical,'checkpoints/local-v8-update92/v8-update92-repair3-continuation-147/after-policy.pt')]
            if update==254:candidates=[(NIGHT/'imports/bcdee6ecc3609f3024f304c15ef8ccba7e605404/handoff/bundle/dependencies/initial-policy.pt').read_bytes()]
            if update==412:candidates=[(NIGHT/'campaign-v1/recoveries/http-transport-003/batches/train-0264-seat0/execution/after-policy.pt').read_bytes()]
            found=next((raw for raw in candidates if hashlib.sha256(raw).hexdigest()==digest),None)
            assert found is not None,'Missing immutable initial policy dependency'
            p=folder/(kind+'-policy.pt');p.write_bytes(found);dependencies[kind]=dict(original_binding=bound,relocated=binding(p))
        backup=BASE/f'independent-checkpoint-backup/update{update}';shutil.copytree(folder,backup)
        result[str(update)]=dict(private_commit=commit,private_path=prefix,files=files,dependencies=dependencies,
            original_source=plan['source'],progress=report['cumulative_progress'],metadata_rewritten=False,backup=str(backup))
    assert result['412']['files']['after-policy.pt']['sha256']=='1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88'
    assert result['412']['files']['after-resume.pt']['sha256']=='09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321'
    return result


def main():
    value=dict(schema='heroes-frozen-transfer-inputs-v1',checkpoints=checkpoints(),maps=maps())
    schedule=[]
    for index in range(16):
        for scenario in ('easy','original'):
            for seat in (0,1):
                models=[146,254,412];rotation=(index+seat+(scenario=='original'))%3;models=models[rotation:]+models[:rotation]
                for update in models:
                    schedule.append(dict(id=f'transfer-{len(schedule)+1:03d}',checkpoint=update,scenario=scenario,layout=index if scenario=='easy' else None,
                        seat=seat,map_key=f'easy-{index}-{seat}' if scenario=='easy' else 'original',
                        engine_seed=(810001 if scenario=='easy' else 830001)+2*index+seat,
                        policy_seed=(820001 if scenario=='easy' else 840001)+2*index+seat))
    value['transfer_schedule']=schedule;assert len(schedule)==192
    value['gate_schedule']=[dict(id=f'gate-{backend}-seat{seat}-{index}',checkpoint=412,backend=backend,seat=seat,
        engine_seed=710001+index,policy_seed=720001+index,map_key=f'gate-{seat}')
        for backend in ('cold-reference','accelerated-embedded') for seat in (0,1) for index in range(2)]
    write(BASE/'inputs.json',value);print(json.dumps(dict(checkpoints=list(value['checkpoints']),maps=len(value['maps']),transfer_games=len(schedule),gate_games=8)))


if __name__=='__main__':main()
