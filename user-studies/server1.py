from aiohttp import web
import socketio
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os.path
from experiments1 import Experiment
import hashlib
import custom_highway_env

experiments = {}

def encode_image(img):
    hex_data = io.BytesIO()
    img.save(hex_data, format='PNG')
    return base64.b64encode(hex_data.getvalue()).decode('ascii')

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

async def index(request):
    path = request._message.path[2:]
    if len(path) < 9 or path[:9].lower() != 'username=':
        filename = 'invalidurl.html'
    else:
        username = path[9:]
        if len(username) == 0:
            filename = 'invalidusername.html'
        elif os.path.exists('results/e1_' + username + '.pkl'):
            with open('results/e1_' + username + '.pkl', 'rb') as f:
                temp_experiment = pickle.load(f)
            if temp_experiment['alreadyCompleted']:
                filename = 'alreadyplayed.html'
            else:
                filename = 'alreadystarted.html'
        else:
            filename = 'index1.html'
    with open(filename) as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('sendAction')
async def receive_action(sid, message):
    img, rew, done, question = experiments[message['username']].step(-1)
    await sio.emit('display', encode_image(img), to=sid)
    if done:
        await sio.emit('roundend_data', {'current_cumulative_reward': 0}, to=sid)
    if experiments[message['username']].alreadyCompleted:
        proof = hashlib.md5(('some_hidden_code_' + message['username']).encode()).hexdigest()
        await sio.emit('experimentover', proof, to=sid)
    if len(question) > 0:
        await sio.emit('create_question', {'options': question}, to=sid)

@sio.on('startSignal')
async def create_env(sid, message):
    experiments[message['username']] = Experiment(message['username'])
    img, _, _, _ = experiments[message['username']].reset()
    await sio.emit('gameIsReady', '', to=sid)
    await sio.emit('display', encode_image(img), to=sid)
    
    
@sio.on('sendAnswer')
async def receive_answer(sid, message):
    img, rew, done, question = experiments[message['username']].step(message['data'])
    await sio.emit('display', encode_image(img), to=sid)
    if done:
        await sio.emit('roundend_data', {'current_cumulative_reward': 0}, to=sid)
    if experiments[message['username']].alreadyCompleted:
        proof = hashlib.md5(('some_hidden_code_' + message['username']).encode()).hexdigest()
        del experiments[message['username']]
        await sio.emit('experimentover', proof, to=sid)
    if len(question) > 0:
        await sio.emit('create_question', {'options': question}, to=sid)


# We bind our aiohttp endpoint to our app router
app.router.add_get('/', index)

# We kick off our server
if __name__ == '__main__':
    web.run_app(app, port=8080)
