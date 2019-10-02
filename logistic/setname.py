import os

def main():
    filepath = "./dataset/train"
    for filename in os.listdir(filepath):
        print(filename)
        if filename == "bird":
            for j, fname in enumerate(os.listdir(os.path.join(filepath, filename))):
                dst = "bird." + str(j) + ".jpg"
                src = os.path.join(os.path.join(filepath, filename),fname)
                dst = os.path.join(os.path.join(filepath, filename), dst)
                os.rename(src, dst)
        else:
            for j, fname in enumerate(os.listdir(os.path.join(filepath, filename))):
                dst = "nonbird." + str(j) + ".jpg"
                src = os.path.join(os.path.join(filepath, filename),fname)
                dst = os.path.join(os.path.join(filepath, filename), dst)
                os.rename(src, dst)

if __name__ == "__main__":
    main()