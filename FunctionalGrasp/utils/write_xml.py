#导入minidom
from xml.dom import minidom

def write_xml(obj_name, r, t, a, path, mode='train'):
    dom=minidom.Document()
    world=dom.createElement('world')
    dom.appendChild(world)
    #  graspableBody
    graspableBody = dom.createElement('graspableBody')
    world.appendChild(graspableBody)

    filename = dom.createElement('filename')
    graspableBody.appendChild(filename)
    if mode=='new':
        name_text=dom.createTextNode('good_shapes/{}.xml'.format(str(obj_name)))
    else:
        name_text=dom.createTextNode('good_shapes/{}_scaled.obj.smoothed.xml'.format(str(obj_name)))
    filename.appendChild(name_text)

    transform = dom.createElement('transform')
    graspableBody.appendChild(transform)

    fullTransform = dom.createElement('fullTransform')
    transform.appendChild(fullTransform)
    fullTransform_text=dom.createTextNode('(+1 +0 +0 +0)[+0 +0 +0]')
    fullTransform.appendChild(fullTransform_text)

    # robot
    robot = dom.createElement('robot')
    world.appendChild(robot)

    filename_r = dom.createElement('filename')
    robot.appendChild(filename_r)
    name_text_r = dom.createTextNode('models/robots/Barrett/Barrett.xml')
    filename_r.appendChild(name_text_r)

    dofValues = dom.createElement('dofValues')
    robot.appendChild(dofValues)
    name_text_rd = dom.createTextNode('{} {} 0 0 {} 0 0 {} 0 0'.format(str(a[0]), str(a[1]), str(a[2]), str(a[3])))
    dofValues.appendChild(name_text_rd)

    transform_r = dom.createElement('transform')
    robot.appendChild(transform_r)
    fullTransform_r = dom.createElement('fullTransform')
    transform_r.appendChild(fullTransform_r)
    fullTransform_text_r=dom.createTextNode('({} {} {} {})[{} {} {}]'.format(str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(t[0]), str(t[1]), str(t[2])))  # +3,+11在组装ros用模型后可以不要
    fullTransform_r.appendChild(fullTransform_text_r)

    # camera
    camera = dom.createElement('camera')
    world.appendChild(camera)

    position = dom.createElement('position')
    camera.appendChild(position)
    name_text_c_1=dom.createTextNode('-2.21912 -6.21883 +479.86')
    position.appendChild(name_text_c_1)

    orientation = dom.createElement('orientation')
    camera.appendChild(orientation)
    name_text_c_2=dom.createTextNode('+0 +0 +0 +1')
    orientation.appendChild(name_text_c_2)

    focalDistance = dom.createElement('focalDistance')
    camera.appendChild(focalDistance)
    name_text_c_3=dom.createTextNode('+300.068')
    focalDistance.appendChild(name_text_c_3)

    try:
        with open(path,'w') as fh:
            # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
            # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
            dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='utf-8')
            # print('写入xml OK!')
    except Exception as err:
        print('错误信息：{0}'.format(err))
