'''
1. copy grasp results (all .xml file) to '/home/lm/graspit/worlds'
2. python show_result.py
'''
import os
import sys
sys.path.append("")

root_dir = '/home/lm/graspit'  # replace it with your graspit root dir
os.environ['GRASPIT'] = root_dir

if __name__ == '__main__':
    world_path = root_dir + '/worlds'
    sub_path = []
    sub_path.append('')

    for sub_name in sorted(os.listdir(world_path)):
        if sub_name.endswith('.xml'):
            xml_name = os.path.splitext(sub_name)[0]
            os.system(root_dir + '/build/graspit_simulator --world {}'.format(xml_name))
