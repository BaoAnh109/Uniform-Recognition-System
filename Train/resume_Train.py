from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\NCKH_AI\Test\Vui\runs\detect\School_Uniform_Project\Retrain_Full_Model_100epochs2\weights\last.pt")
    model.train(resume=True)