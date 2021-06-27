"""
MicroPython script for Sipeed camera
"""
import sensor, image, lcd, time
import KPU as kpu

lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((128, 128))
sensor.set_vflip(1)
sensor.run(1)
lcd.clear()

f=open('labels.txt','r')
labels=f.readlines()
f.close()

print(kpu.memtest())
latency_result = None
fps_result = None

def inference(model_file):
	task = kpu.load(model_file)
	kpu.set_outputs(task, 0, 1, 1, 2)
	clock = time.clock()
	while(True):
	    img = sensor.snapshot()
	    clock.tick()
	    fmap = kpu.forward(task, img)
	    fps=clock.fps()
	    plist=fmap[:]
	    pmax=max(plist)
	    max_index=plist.index(pmax)
	    a = lcd.display(img, oft=(0,0))
	    lcd.draw_string(0, 128, "%.2f:%s                            "%(pmax, labels[max_index].strip()))
	_ = kpu.deinit(task)


def measure_fps(model_file):
	task = kpu.load(model_file)
	kpu.set_outputs(task, 0, 1, 1, 2)
	clock = time.clock()
	fps_ = []
	for i in range(20):
		img = sensor.snapshot()
		clock.tick()
		fmap = kpu.forward(task, img)
		lcd.display(img, oft=(0,0))
		fps_.append(clock.fps())
	average_fps = sum(fps_) / len(fps_)
	print(average_fps)
	global fps_result
	fps_result = average_fps
	_ = kpu.deinit(task)


def query_fps():
	print(fps_result)


def measure_latency(model_file):
	task = kpu.load(model_file)
	kpu.set_outputs(task, 0, 1, 1, 2)
	clock = time.clock()
	latency_ = []
	for i in range(20):
		img = sensor.snapshot()
		clock.tick()
		t1 = time.ticks_us()   
		fmap = kpu.forward(task, img)
		t2 = time.ticks_diff(time.ticks_us(), t1) /1000
		lcd.display(img, oft=(0,0))
		latency_.append(t2)
	average_latency = sum(latency_) / len(latency_)
	print(average_latency)
	global latency_result
	latency_result = average_latency
	_ = kpu.deinit(task)


def query_latency():
	print(latency_result)



















	

