import tensorflow as tf
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

# 指定 events.out.tfevents 文件的路径
file_path = 'log/image/1/events.out.tfevents.1678265334.server-mercury'

# 创建 EventFileLoader 对象读取文件
loader = EventFileLoader(file_path)

for event in loader.Load():
    # Check if the event is a summary event and has a "scalar" value
    if event.WhichOneof('what') == 'summary' and len(event.summary.value) > 0:
        # Iterate through the summary values and print them
        for value in event.summary.value:
            print(value.tag, value.simple_value)
            
# # 遍历文件中的每个事件
# for event in loader.Load():
#     # 判断当前的事件是否为损失或精度数据并且存在对应的 summary 值
#     if event.summary and len(event.summary.value) > 0:
#         step = event.step    # 获取当前步骤数
#         for value in event.summary.value:
#             if value.tag == 'loss':   # 如果是损失值数据
#                 loss = value.simple_value   # 获取损失值
#             elif value.tag == 'accuracy':   # 如果是精度数据
#                 accuracy = value.simple_value  # 获取精度值
#         print('Step:', step, 'Loss:', loss, 'Accuracy:', accuracy)
