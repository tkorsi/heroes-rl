"""Authored initial-state curriculum; native rules and actions are unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile
from experiments.explorer.create_capability_scenarios import ordinary_payloads


def payloads(stage=0, seat=0):
    if stage not in (0, 1, 2, 3) or seat not in (0, 1):
        raise ValueError('Unknown curriculum stage or seat')
    value = ordinary_payloads()
    if stage == 2:
        return value
    color = ('red', 'blue')[seat]
    opponent = 1-seat
    objects = value['objects.json']
    hero = next(x for x in objects if x['instanceName'] == f'hero_{seat}')
    town = next(x for x in objects if x['instanceName'] == f'town_{opponent}')
    # A town's visitable cell is two columns left of its sprite anchor.
    x, y = town['x']-2, town['y']
    hero.update(x=x+1, y=y+(1 if stage == 0 else 4))
    hero['options']['army'] = [{'type':'core:pikeman','amount':100 if stage == 0 else 50}]
    value['objects.json'] = [x for x in objects if x['instanceName'] not in
                            (f'town_{seat}', f'hero_{opponent}')]
    if stage == 3:
        # Smaller open authored board, without impassable corridors or filtered actions.
        value['objects.json'] = [hero,town]
        town.update(x=6,y=5);hero.update(x=5,y=6)
        hero['options']['army']=[{'type':'core:pikeman','amount':100}]
        value['surface_terrain.json']=[row[:8] for row in value['surface_terrain.json'][:8]]
        value['header.json']['mapLevels']['surface'].update(width=8,height=8)
        value['header.json']['players'][('red','blue')[opponent]]['mainTown'].update(x=4,y=5)
    players = value['header.json']['players']
    players[color].pop('mainTown',None)
    other = players[('red','blue')[opponent]]
    other.pop('mainHero',None);other['heroes']={}
    value['header.json']['name'] = f'Curriculum v1 stage{stage} learner {color}'
    value['header.json']['description'] = ('Declared initial-state distribution: townless learner near opponent town; '
        'opponent starts without a hero. Native recruitment, fog, actions, battles and elimination rules unchanged.')
    return value


def create(path, stage, seat):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in payloads(stage,seat).items():
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,json.dumps(data,sort_keys=True,indent=2)+'\n')
    return path


def verify(path,stage,seat):
    with zipfile.ZipFile(path) as archive:
        actual={name:json.loads(archive.read(name)) for name in archive.namelist()}
    if actual != payloads(stage,seat):raise ValueError('Curriculum map differs from authored state')
    return True
