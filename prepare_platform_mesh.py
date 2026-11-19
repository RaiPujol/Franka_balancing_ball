"""Generate the revision 2 tray-frame STL or verify its matching 3MF export."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial import cKDTree

from stl_scene import SOURCE, SOURCE_HASH, LOCAL_MESH, LOCAL_HASH

TRIANGLE = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
# Source coordinates, in mm. The square extends from y=21 to y=254.
TOP_CENTER_MM = np.array([116.5, 137.5, 21.0])


def prepare_asset():
    raw = SOURCE.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH or len(raw) != 84 + count * TRIANGLE.itemsize:
        raise ValueError('Source is not the inspected binary STL')
    triangles = np.frombuffer(raw, offset=84, dtype=TRIANGLE).copy()
    source_vertices = triangles['v'].astype(float)
    triangles['v'] -= TOP_CENTER_MM
    asset = raw[:84] + triangles.tobytes()
    if hashlib.sha256(asset).hexdigest() != LOCAL_HASH:
        raise ValueError('Translated mesh differs from the verified asset')
    return source_vertices, asset


def oriented_faces(faces):
    # Treat cyclic rotations of the same three indices as the same face.
    return Counter(tuple(np.roll(face, -np.argmin(face))) for face in faces)


def verify_3mf(source_triangles, project_path):
    """Compare this Bambu project's object mesh in CAD coordinates.

    Its build transform places the part on the printer bed. The source_offset
    metadata restores the CAD coordinates of the object before that operation.
    The component transform for this export is identity.
    """
    with zipfile.ZipFile(project_path) as archive:
        doc = ET.fromstring(archive.read('3D/3dmodel.model'))
        model = ET.fromstring(archive.read('3D/Objects/object_8.model'))
        settings = ET.fromstring(archive.read('Metadata/model_settings.config'))
    component = doc.find('.//{*}component')
    identity = [1,0,0,0,1,0,0,0,1,0,0,0]
    if (doc.get('unit') != 'millimeter' or model.get('unit') != 'millimeter' or component is None
            or not np.allclose(np.fromstring(component.attrib['transform'], sep=' '), identity)):
        raise ValueError('Unexpected project units or component transform')
    part = settings.find('.//part')
    properties = {entry.attrib['key']: entry.attrib['value'] for entry in part.findall('metadata')}
    source_offset = np.array([float(properties['source_offset_' + axis]) for axis in 'xyz'])
    vertices = np.array([[float(v.attrib[a]) for a in 'xyz'] for v in model.findall('.//{*}vertex')])
    faces = np.array([[int(t.attrib[k]) for k in ('v1','v2','v3')] for t in model.findall('.//{*}triangle')])
    distances, indices = cKDTree(vertices + source_offset).query(source_triangles)
    if distances.max() > 1e-4 or oriented_faces(indices) != oriented_faces(faces):
        raise ValueError('3MF object mesh differs from the STL')
    return len(faces), float(distances.max())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-3mf', type=Path, help='path to the original Bambu Studio project')
    options = parser.parse_args()
    triangles, asset = prepare_asset()
    if options.verify_3mf:
        count, error = verify_3mf(triangles, options.verify_3mf)
        print(f'3MF matches all {count} oriented STL triangles, max vertex distance {error:.8f} mm')
    LOCAL_MESH.write_bytes(asset)
    print(f'Wrote verified tray-frame STL: {LOCAL_MESH.name}')
