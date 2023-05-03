from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
import sys

class AnimatedGif(tk.Label):
    def __init__(self, master, filename, duration, **kwargs):
        self.duration = duration
        self.frames = []
        self.idx = 0
        self.paused = False
        self.speed = 1.0

        im = Image.open(filename)
        self.width, self.height = im.size

        for frame in ImageSequence.Iterator(im):
            self.frames.append(ImageTk.PhotoImage(frame))

        super().__init__(master, image=self.frames[0], **kwargs)

        self.after(0, self.animate)

    def animate(self):
        if not self.paused:
            self.idx = (self.idx + 1) % len(self.frames)
            self.configure(image=self.frames[self.idx])
            self.after(int(self.duration/len(self.frames)/self.speed), self.animate)

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def set_speed(self, speed):
        self.speed = speed

    def reset(self):
        self.idx = 0
        self.paused = False
        self.speed = 1.0
        self.configure(image=self.frames[0])

if __name__ == '__main__':
    root = tk.Tk()

    anim = AnimatedGif(root, sys.argv[1], 10000)
    anim.pack()

    def pause_unpause():
        if anim.paused:
            anim.paused = False
            anim.after(0, anim.animate)
        else:
            anim.paused = True

    def faster():
        anim.set_speed(anim.speed*2)

    def slower():
        anim.set_speed(anim.speed/2)

    def reset():
        anim.reset()

    button_frame = tk.Frame(root)
    button_frame.pack()

    pause_button = tk.Button(button_frame, text='Pause/Unpause', command=pause_unpause)
    pause_button.pack(side=tk.LEFT)

    faster_button = tk.Button(button_frame, text='Faster', command=faster)
    faster_button.pack(side=tk.LEFT)

    slower_button = tk.Button(button_frame, text='Slower', command=slower)
    slower_button.pack(side=tk.LEFT)

    reset_button = tk.Button(button_frame, text='Reset', command=reset)
    reset_button.pack(side=tk.LEFT)

    root.mainloop()
