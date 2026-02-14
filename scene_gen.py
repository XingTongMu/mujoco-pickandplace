import random
import os

class SceneGenerator:
    def __init__(self, template_path="template_scene.xml"):
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
    def generate_random_scene(self, output_path="temp_scene.xml"):
        # 随机参数生成
        obj_type = random.choice(["box", "sphere", "cylinder", "capsule"])
        
        # 随机颜色 RGBA
        color = [random.random(), random.random(), random.random(), 1.0]
        color_str = f"{color[0]:.2f} {color[1]:.2f} {color[2]:.2f} 1"
        
        # 随机位置 (x: 0.4~0.6, y: -0.1~0.1)
        pos_x = 0.4 + random.random() * 0.2
        pos_y = -0.1 + random.random() * 0.2
        
        geom_xml = ""
        size_info = []
        
        # 基础尺寸
        base_size = 0.02 + random.random() * 0.02 # 0.02 ~ 0.04
        
        if obj_type == "box":
            # size: half-width, half-depth, half-height
            sx = base_size
            sy = base_size
            sz = base_size
            size_str = f"{sx:.3f} {sy:.3f} {sz:.3f}"
            pos_z = sz # 放在地面上
            geom_xml = f'<geom type="box" size="{size_str}" rgba="{color_str}" mass="0.1" friction="1 0.005 0.0001"/>'
            size_info = [sx, sy, sz]
            
        elif obj_type == "sphere":
            # size: radius
            r = base_size
            size_str = f"{r:.3f}"
            pos_z = r
            geom_xml = f'<geom type="sphere" size="{size_str}" rgba="{color_str}" mass="0.1" friction="1 0.005 0.0001"/>'
            size_info = [r]
            
        elif obj_type == "cylinder":
            # size: radius, half-height
            r = base_size
            h = base_size
            size_str = f"{r:.3f} {h:.3f}"
            pos_z = h
            geom_xml = f'<geom type="cylinder" size="{size_str}" rgba="{color_str}" mass="0.1" friction="1 0.005 0.0001"/>'
            size_info = [r, h]
            
        elif obj_type == "capsule":
            # size: radius, half-length
            r = base_size * 0.8
            h = base_size
            size_str = f"{r:.3f} {h:.3f}"
            pos_z = r # 横放时的半径高度，或者竖放
            # 简单起见，这里假设竖放或者横放，胶囊体默认是Z轴对齐
            # 为了好抓，我们让它立着
            pos_z = h + r
            geom_xml = f'<geom type="capsule" size="{size_str}" rgba="{color_str}" mass="0.1" friction="1 0.005 0.0001"/>'
            size_info = [r, h]

        pos_str = f"{pos_x:.3f} {pos_y:.3f} {pos_z:.3f}"
        
        # 构建物体 Body XML
        obj_xml = f'''
    <body name="target_object" pos="{pos_str}">
      <freejoint/>
      {geom_xml}
      <site name="object_site" pos="0 0 0" size="0.005" rgba="1 1 1 0.3"/>
    </body>
        '''
        
        # 替换模板
        scene_content = self.template.replace("<!-- RANDOM_OBJECT_PLACEHOLDER -->", obj_xml)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(scene_content)
            
        return {
            "type": obj_type,
            "size": size_info,
            "pos": [pos_x, pos_y, pos_z],
            "color": color,
            "xml_path": output_path
        }

if __name__ == "__main__":
    gen = SceneGenerator()
    info = gen.generate_random_scene()
    print("Generated:", info)
