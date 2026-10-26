"""Authored fixture geometry only; this data is never a policy input or mask."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def validate(hero_anchor,town_anchor):
    hx,hy=hero_anchor;tx,ty=town_anchor
    # CGHeroInstance::convertToVisitablePos subtracts the native hero template's
    # (1,0,0) visitable offset; the castle template subtracts (2,0,0).
    start=(hx-1,hy);target=(tx-2,ty)
    result=dict(hero_anchor=[hx,hy],hero_visitable=list(start),town_anchor=[tx,ty],town_visitable=list(target),valid=False,policy_input=False)
    if not (0<=start[0]<8 and 0<=start[1]<8 and 4<=tx<8 and 2<=ty<7):
        return dict(result,reason='native_visitable_or_town_footprint_out_of_bounds')
    blocked={(x,y) for x in range(tx-4,tx+1) for y in range(ty-2,ty+1)}-{target}
    if start in blocked or start==target:return dict(result,reason='initial_hero_collides_with_town')
    seen={start};queue=[start]
    while queue:
        x,y=queue.pop()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt=(x+dx,y+dy)
            if 0<=nxt[0]<8 and 0<=nxt[1]<8 and nxt not in seen and nxt not in blocked:seen.add(nxt);queue.append(nxt)
    return dict(result,valid=target in seen,reason='valid_conservative_native_geometry' if target in seen else 'no_open_orthogonal_connection')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());rows=[]
    for key,bound in manifest['maps'].items():
        if not key.startswith('easy-'):continue
        path=Path(bound['path']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=bound['sha256']:raise ValueError('Frozen map changed')
        with zipfile.ZipFile(path) as z:
            objects=json.loads(z.read('objects.json'));terrain=json.loads(z.read('surface_terrain.json'))
        if len(objects)!=2 or not all(t.startswith('dt') for row in terrain for t in row):raise ValueError('Undeclared map object/terrain')
        hero=next(o for o in objects if o['type']=='hero');town=next(o for o in objects if o['type']=='town')
        rows.append(dict(map_key=key,map_sha256=bound['sha256'],**validate([hero['x'],hero['y']],[town['x'],town['y']])))
    output=dict(native_anchor_offset_accounted=True,policy_input=False,maps_or_schedule_rewritten=False,maps=rows,
        valid_maps=sum(r['valid'] for r in rows),invalid_maps=sum(not r['valid'] for r in rows))
    if a.output.exists():raise ValueError('Preserve previous geometry audit')
    a.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
