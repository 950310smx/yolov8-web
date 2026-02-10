from PIL import Image
print("Pillow导入成功！")

# 创建一个简单的测试图像
img = Image.new('RGB', (100, 100), color='red')
img.save('test.png')
print("图像创建和保存成功！")