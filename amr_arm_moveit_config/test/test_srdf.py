"""El SRDF es coherente con la URDF de amr_description y con arm_server.yaml.

No necesita MoveIt: sirve para detectar nombres de links/juntas que cambien
al sustituir la URDF por el export de SolidWorks.
"""
import os
import xml.etree.ElementTree as ET

import pytest
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def urdf():
    path = os.path.join(get_package_share_directory('amr_description'), 'urdf', 'amr.urdf.xacro')
    return ET.fromstring(xacro.process_file(path).toxml())


@pytest.fixture(scope='module')
def srdf():
    return ET.parse(os.path.join(PKG, 'config', 'omrover_mm.srdf')).getroot()


def links(urdf):
    return {e.get('name') for e in urdf.findall('link')}


def joints(urdf):
    return {e.get('name'): e for e in urdf.findall('joint')}


def test_names_exist(urdf, srdf):
    ln, jn = links(urdf), joints(urdf)
    for e in srdf.iter():
        if e.tag in ('link',) and e.get('name'):
            assert e.get('name') in ln, e.get('name')
        if e.tag == 'chain':
            assert e.get('base_link') in ln and e.get('tip_link') in ln
        if e.tag == 'joint' and e.get('name'):
            assert e.get('name') in jn, e.get('name')
        if e.tag == 'disable_collisions':
            assert e.get('link1') in ln and e.get('link2') in ln
        if e.tag == 'end_effector':
            assert e.get('parent_link') in ln


def test_chain_is_connected(urdf, srdf):
    chain = srdf.find("group[@name='arm']/chain")
    parent = {j.find('child').get('link'): j for j in urdf.findall('joint')}
    link, movable = chain.get('tip_link'), []
    while link != chain.get('base_link'):
        assert link in parent, f'{link} no desciende de {chain.get("base_link")}'
        j = parent[link]
        if j.get('type') != 'fixed':
            movable.append(j.get('name'))
        link = j.find('parent').get('link')
    assert sorted(movable) == ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4']


def test_group_states_within_limits(urdf, srdf):
    jn = joints(urdf)
    for gs in srdf.findall('group_state'):
        for j in gs.findall('joint'):
            lim = jn[j.get('name')].find('limit')
            v = float(j.get('value'))
            assert float(lim.get('lower')) - 1e-6 <= v <= float(lim.get('upper')) + 1e-6, \
                f"{gs.get('name')}: {j.get('name')}={v}"


def test_poses_match_arm_server(srdf):
    path = os.path.join(get_package_share_directory('amr_arm_control'), 'config',
                        'arm_server.yaml')
    with open(path) as f:
        poses = yaml.safe_load(f)['arm_server']['ros__parameters']['poses']
    for name, q in poses.items():
        gs = srdf.find(f"group_state[@name='{name}']")
        assert gs is not None, f'falta la pose {name} en el SRDF'
        vals = {j.get('name'): float(j.get('value')) for j in gs.findall('joint')}
        assert [vals[f'arm_joint_{i}'] for i in range(1, 5)] == pytest.approx(q, abs=1e-4)


def test_controller_limits_consistent():
    with open(os.path.join(PKG, 'config', 'joint_limits.yaml')) as f:
        lim = yaml.safe_load(f)['joint_limits']
    path = os.path.join(get_package_share_directory('amr_arm_control'), 'config',
                        'arm_controller.yaml')
    with open(path) as f:
        a_max = yaml.safe_load(f)['arm_controller']['ros__parameters']['max_accel']
    for i in range(1, 5):
        assert lim[f'arm_joint_{i}']['max_acceleration'] <= a_max + 1e-9
