"""Deterministic procedural images. These are NOT diffusion samples or human annotations."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'data/processed/diagnostics-v1'
    out.mkdir(parents=True, exist_ok=False)
    scenes = [
        ('color', [('square','red',(100,120,220,240))],
         'A red square on a white background.', ['The square is red.'],
         'A blue square on a white background.', ['The square is blue.']),
        ('count', [('circle','blue',(40,130,120,210)),('circle','blue',(170,130,250,210)),('circle','blue',(300,130,380,210))],
         'Exactly three blue circles.', ['There are exactly three circles.', 'All circles are blue.'],
         'Exactly two blue circles.', ['There are exactly two circles.', 'All circles are blue.']),
        ('position', [('circle','red',(40,130,120,210)),('square','blue',(270,130,350,210))],
         'A red circle to the left of a blue square.', ['The circle is red.', 'The square is blue.', 'The circle is left of the square.'],
         'A red circle to the right of a blue square.', ['The circle is red.', 'The square is blue.', 'The circle is right of the square.']),
        ('binding', [('circle','green',(50,130,150,230)),('square','yellow',(270,130,370,230))],
         'A green circle and a yellow square.', ['The circle is green.', 'The square is yellow.'],
         'A yellow circle and a green square.', ['The circle is yellow.', 'The square is green.'])]
    rows = []
    for category, objects, positive, positive_claims, negative, negative_claims in scenes:
        image = Image.new('RGB', (448, 336), 'white')
        draw = ImageDraw.Draw(image)
        for shape, color, box in objects:
            (draw.ellipse if shape == 'circle' else draw.rectangle)(box, fill=color)
        path = out/(category+'.png')
        image.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for suffix, request, claims, expected in [('positive',positive,positive_claims,True),('negative',negative,negative_claims,False)]:
            rows.append({'id':category+'-'+suffix,'category':category,'image_group':category,
                         'image':str(path.relative_to(ROOT)), 'image_sha256':digest,
                         'instruction':request,'claims':claims,'expected':expected,
                         'source':'procedural-diagnostic-v1','split':'development-only',
                         'reference_type':'programmatic scene specification, not human rating'})
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('DIAGNOSTICS_READY',len(rows),'cases,',len(scenes),'independent images',flush=True)


if __name__ == '__main__':
    main()
