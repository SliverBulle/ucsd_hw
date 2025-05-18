from PIL import Image

# 打开四张 PNG 图片
image1 = Image.open("data/1130/918.png")
image2 = Image.open("data/1130/198.png")
image3 = Image.open("data/1130/213.png")
image4 = Image.open("data/1130/2321.png")

# 获取每张图片的宽度和高度
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size
width4, height4 = image4.size

# 计算拼接后图片的宽度和高度
max_width = max(width1, width2, width3, width4)  # 最大宽度
total_height = height1 + height2 + height3 + height4  # 总高度为四张图片高度之和

# 创建一个新的空白图片，模式为 RGBA（包含透明度）
new_image = Image.new("RGBA", (max_width, total_height))

# 将四张图片粘贴到新图片上
new_image.paste(image1, (0, 0))          # 第一张在最上面
new_image.paste(image2, (0, height1))    # 第二张在第一张下方
new_image.paste(image3, (0, height1 + height2))  # 第三张在第二张下方
new_image.paste(image4, (0, height1 + height2 + height3))  # 第四张在第三张下方

# 保存拼接后的图片
new_image.save("hl_hw5.png")
