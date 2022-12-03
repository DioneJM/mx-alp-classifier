from fastai.vision.all import load_learner
import gradio as gr

learn = load_learner('model.pkl')

categories = ('Cherry switch', 'Alps switch')


def classify_image(img):
    prediction, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

examples = ['u4t.jpg', 'alps.jpg']

iface = gr.Interface(fn=classify_image, inputs=image, outputs=label)
iface.launch()
