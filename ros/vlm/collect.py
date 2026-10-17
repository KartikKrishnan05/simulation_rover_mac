#!/usr/bin/env python3
"""Episoden sammeln und auswerten – mit VLM oder Orakel (Ground Truth).

  pixi run collect find   --episodes 6                      # VLM, jede Taschenposition einmal
  pixi run collect find   --episodes 30 --perception oracle # Experten-Daten, schnell
  pixi run collect goto   --episodes 6
  pixi run collect follow --episodes 1

Pro Episode wird die Szene vorbereitet (Tasche versetzen, Person an den
Start), der Rover an die Startpose gesetzt (Gazebo + AMCL), dann führt der
Agent eine zufällig formulierte Anweisung aus. Erfolg wird mit der Ground
Truth aus Gazebo geprüft. Ergebnisse: data/episodes/results.jsonl und eine
Tabelle am Ende.

Mit --perception vlm interpretiert das VLM auch die Anweisung; im Orakel-
Modus ist die Aufgabe vorgegeben und Räume werden nach Entfernung abgesucht.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import Agent  # noqa: E402
from common import ROOT, in_rect  # noqa: E402
from vlm_client import DEFAULT_MODEL  # noqa: E402

ROOM_WORDS = {'open_space': 'open-plan office', 'lounge': 'lounge', 'meeting_room': 'meeting room',
              'office_a': 'office A', 'office_b': 'office B', 'kitchen': 'kitchen'}
GOTO_TEMPLATES = ['go to the {r}', 'drive to the {r}', 'please take me to the {r}',
                  'head over to the {r}', 'navigate to the {r}']
FIND_TEMPLATES = ['find the blue bag', 'search for the blue backpack', 'where is the blue bag? go there',
                  'look for a blue bag and drive to it', 'bring me to the blue rucksack']
FOLLOW_TEMPLATES = ['follow the person', 'follow that man', 'go after the person walking around',
                    'keep following the person']
PARK_BAG = (-9.2, -4.0)  # beim Folgen aus dem Weg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task', choices=('goto', 'find', 'follow'))
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--perception', choices=('vlm', 'oracle'), default='vlm')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-reset', action='store_true',
                        help='Rover nicht vor jeder Episode an die Startpose setzen')
    parser.add_argument('--no-record', action='store_true')
    parser.add_argument('--follow-time', type=float, default=400.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    agent = Agent(args.perception, args.model, record=not args.no_record)
    sm, results = agent.sm, []
    out = ROOT / 'data' / 'episodes'
    out.mkdir(parents=True, exist_ok=True)
    try:
        agent.wait_ready()
        spots = list(sm['bag_spots'].items())
        rooms = list(sm['rooms'])
        rng.shuffle(spots)
        rng.shuffle(rooms)
        for i in range(args.episodes):
            if not args.no_reset or args.task == 'follow':
                agent.say('setze Rover an die Startpose')
                agent.reset_rover()
            meta = {'task': args.task, 'collect_seed': args.seed, 'episode_index': i}
            plan = None
            if args.task == 'goto':
                room = rooms[i % len(rooms)]
                instruction = rng.choice(GOTO_TEMPLATES).format(r=ROOM_WORDS[room])
                meta['target_room'] = room
                if args.perception == 'oracle':
                    plan = {'task': 'goto', 'room': room, 'target': None}

                def evaluate(room=room):
                    gt = agent.gt.get('rover')
                    return {'gt_success': bool(gt and in_rect(gt[0], gt[1], sm['rooms'][room]['rect'])),
                            'gt_rover': gt}
            elif args.task == 'find':
                spot, (bx, by) = spots[i % len(spots)]
                agent.sim.set_pose('blue_bag', bx, by, rng.uniform(-math.pi, math.pi), z=0.005)
                instruction = rng.choice(FIND_TEMPLATES)
                meta.update(bag_spot=spot, bag_xy=[bx, by])
                if args.perception == 'oracle':
                    plan = {'task': 'find', 'room': None, 'target': 'blue bag'}

                def evaluate(bx=bx, by=by):
                    gt, bag = agent.gt.get('rover'), agent.gt.get('bag') or (bx, by)
                    d = math.hypot(gt[0] - bag[0], gt[1] - bag[1]) if gt else math.inf
                    return {'gt_success': d < 2.2, 'gt_distance_to_bag': round(d, 2)}
            else:
                agent.sim.set_pose('blue_bag', *PARK_BAG, 0.0, z=0.005)
                agent.pub_person_cmd.publish(String(data='reset'))
                time.sleep(2.0)
                instruction = rng.choice(FOLLOW_TEMPLATES)
                if args.perception == 'oracle':
                    plan = {'task': 'follow', 'room': None, 'target': 'person'}

                def evaluate():
                    return {'gt_success': agent.person_state == 'done', 'person_state': agent.person_state}
            print(f'\n=== Episode {i + 1}/{args.episodes}: "{instruction}"', flush=True)
            time.sleep(1.0)
            res = agent.run(instruction, plan=plan, meta=meta, follow_time=args.follow_time,
                            start_person=args.task == 'follow', evaluate=evaluate)
            res.update(instruction=instruction, **meta, perception=args.perception,
                       model=args.model if args.perception == 'vlm' else None)
            results.append(res)
            with open(out / 'results.jsonl', 'a') as f:
                f.write(json.dumps(res, default=str) + '\n')
            time.sleep(2.0)  # Recorder schreibt die Episode
    except KeyboardInterrupt:
        pass
    finally:
        agent.shutdown()

    if results:
        print('\n#   Erfolg(Agent) Erfolg(GT)  VLM-Aufrufe  VLM-s  Dauer-s  Meldung')
        for i, r in enumerate(results):
            print(f'{i + 1:<3} {str(r["success"]):<13} {str(r.get("gt_success")):<11} '
                  f'{r["vlm_calls"]:<12} {r["vlm_seconds"]:<6} {r["wall_seconds"]:<8} {r["message"]}')
        ok = sum(bool(r.get('gt_success')) for r in results)
        print(f'\nErfolgsquote (Ground Truth): {ok}/{len(results)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
