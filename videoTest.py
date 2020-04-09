from imageai.Detection import VideoObjectDetection
import os

def forEachFrame(frame_number, output_array, output_count):
	print("For Frame ", frame_number)
	print("Output for each object : ", output_array)
	print("Output count for unique objects : ", output_count)
	print("----------------------------------------------")

exec_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exec_path, "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
	input_file_path=os.path.join(exec_path, "car.mp4"),
	output_file_path=os.path.join(exec_path, "NewVideoTest"),
	frames_per_second=30,
	log_progress=True,
	per_frame_function=forEachFrame,
	minimum_percentage_probability=30)

print(video_path)

