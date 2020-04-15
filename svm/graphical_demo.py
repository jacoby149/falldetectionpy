from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import animation as animation

def visualize(magnitudes, prediction):

    # plot time vs magnitude
    intervals = [0.1 * x for x in range(len(magnitudes))]
    fig = plt.figure()
    plt.plot(intervals, magnitudes)
    plt.xlabel("time (s)")
    plt.ylabel("acceleration")
    ax = plt.gca()

    top = max(magnitudes) # top position
    bottom = min(magnitudes)
    width = len(magnitudes) // WINDOW_SIZE

    rectangles = []
    # break up data into sliding windows and add rectangles for each sliding window
    for i in range(0, len(magnitudes)-WINDOW_SIZE):
        rectangles.append(patches.Rectangle((i/WINDOW_SIZE, bottom), 1, top-bottom,
                          linewidth=1, edgecolor='r', facecolor='none', zorder=3))

    # store texts for each prediction
    texts = []
    for p in prediction:
        if p == 1:
            texts.append("fall")
        else:
            texts.append("normal")

    # draw first rectangle and prediction
    ax.add_patch(rectangles[0])
    ax.set_title(texts[0])

    running = True
    # updates the slider
    def update(i):
        if len(rectangles) > 1:
            rectangles[0].remove()
            rectangles.pop(0)
            ax.add_patch(rectangles[0])
            texts.pop(0)
            ax.set_title(texts[0])

    # starts the animation on mouse click
    def onClick(event):
        nonlocal running
        if running:
            ani.event_source.stop()
            running = False
        else:
            ani.event_source.start()
            running = True

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = animation.FuncAnimation(fig, update, interval=30)
    plt.show()