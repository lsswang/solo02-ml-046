import json
from pathlib import Path
from mapmodel import update_independent
from routing import shortest

if __name__=='__main__':
    cases=json.loads((Path(__file__).parent/'fixtures/cases.json').read_text())
    print(json.dumps({'map':update_independent(**cases['ordinary_map']),
                     'route':shortest(cases['graph'],0,2)},indent=2))
