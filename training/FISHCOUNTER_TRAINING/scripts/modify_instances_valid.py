import json

# Path to annotation file
json_path = 'C:\ITB\KP\kp\github\fishcounter\training\FISHCOUNTER_TRAINING\patin-dataset\annotations\instances_valid.json'


with open(json_path, 'r+', encoding='utf-8') as f:
    data = json.load(f)
    changed = False
    if 'info' not in data:
        data['info'] = {}
        changed = True
    if 'licenses' not in data:
        data['licenses'] = []
        changed = True
    if changed:
        keys = ['info', 'licenses', 'categories', 'annotations', 'images']
        new_data = {k: data[k] for k in keys if k in data}
        f.seek(0)
        json.dump(new_data, f, separators=(',', ':'))  # minified output
        f.truncate()
        print('Patched:', json_path)
    else:
        print('No patch needed.')