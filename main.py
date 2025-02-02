import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
import json
from collections import deque
import random
import math
import matplotlib.pyplot as plt
import os
import heapq
import webbrowser
from io import BytesIO

# Biblioteca pentru generarea codurilor de bare
import barcode
from barcode.writer import ImageWriter

# Biblioteca pentru decodarea codurilor de bare
from pyzbar.pyzbar import decode
from PIL import Image as PILImage, ImageDraw, ImageFont

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Configurare TensorFlow pe GPU/CPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
cpu_count = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)

##############################################################################
# DEFINIRE DIRECȚII + FUNCȚII RUTARE
##############################################################################
ACTIONS_8DIR = [
    (0, -1),  # N
    (0, 1),   # S
    (-1, 0),  # W
    (1, 0),   # E
    (-1, -1), # NW
    (1, -1),  # NE
    (-1, 1),  # SW
    (1, 1),   # SE
]
NUM_ACTIONS = len(ACTIONS_8DIR)

def compute_route_bfs_8dir(factory_map, start, warehouse):
    if start is None or warehouse is None:
        return []
    directions = ACTIONS_8DIR
    w, h = factory_map.width, factory_map.height
    visited = [[False]*w for _ in range(h)]
    parent = {}
    queue = deque()
    sx, sy = start
    wx, wy = warehouse

    queue.append((sx, sy))
    visited[sy][sx] = True
    parent[(sx, sy)] = None
    found = False

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == (wx, wy):
            found = True
            break

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if not visited[ny][nx] and not factory_map.is_obstacle(nx, ny):
                    visited[ny][nx] = True
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    if not found:
        return []

    path = []
    cur = (wx, wy)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def compute_route_qlearning_8dir(factory_map, start, warehouse,
                                 episodes=1000, alpha=0.1, gamma=0.9,
                                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                                 max_steps=300, reward_goal=10.0,
                                 reward_step=-0.1, reward_obstacle=-1.0):
    """
    Q-learning tabular simplu
    """
    if start is None or warehouse is None:
        return []

    w, h = factory_map.width, factory_map.height
    actions = ACTIONS_8DIR
    num_actions = len(actions)

    Q_table = {}
    def get_q_values(x, y):
        if (x, y) not in Q_table:
            Q_table[(x, y)] = [0.0]*num_actions
        return Q_table[(x, y)]

    def is_valid(nx, ny):
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return False
        return not factory_map.is_obstacle(nx, ny)

    # Training episodes
    for ep in range(episodes):
        x, y = start
        for _ in range(max_steps):
            qv = get_q_values(x, y)
            if random.random() < epsilon:
                a_idx = random.randint(0, num_actions-1)
            else:
                a_idx = max(range(num_actions), key=lambda i: qv[i])

            dx, dy = actions[a_idx]
            nx, ny = x+dx, y+dy
            if not is_valid(nx, ny):
                nx, ny = x, y
                r = reward_obstacle
                done = False
            else:
                if (nx, ny) == warehouse:
                    r = reward_goal
                    done = True
                else:
                    r = reward_step
                    done = False

            old_q = qv[a_idx]
            next_q = get_q_values(nx, ny)
            best_q_next = max(next_q) if not done else 0.0
            target = r + gamma*best_q_next
            qv[a_idx] = old_q + alpha*(target - old_q)

            x, y = nx, ny
            if done:
                break
        epsilon = max(epsilon_min, epsilon*epsilon_decay)

    # Generăm ruta finală
    path = []
    cx, cy = start
    visited = set()
    for _ in range(max_steps):
        path.append((cx, cy))
        if (cx, cy) == warehouse:
            break
        visited.add((cx, cy))
        qv = get_q_values(cx, cy)
        best_a = max(range(num_actions), key=lambda i: qv[i])
        dx, dy = actions[best_a]
        nx, ny = cx+dx, cy+dy
        if not is_valid(nx, ny) or ((nx, ny) in visited):
            break
        cx, cy = nx, ny

    if path and path[-1] != warehouse:
        return []
    return path


def build_heuristic_model():
    inp = keras.Input(shape=(4,))
    x = layers.Dense(32, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_or_load_heuristic_model(model_path="heuristic_model.keras", train_model=False):
    if train_model or not os.path.exists(model_path):
        model = build_heuristic_model()
        X_train = np.random.rand(1000,4).astype(np.float32)
        y_train = np.random.rand(1000,1).astype(np.float32)
        model.fit(X_train,y_train,epochs=5, verbose=0)
        model.save(model_path)
    else:
        model = load_model(model_path)
    return model

def estimate_cost(model, state_tensor):
    cost = model.predict(state_tensor, verbose=0)
    return cost[0][0]

def compute_route_neural_dstar(factory_map, start, goal, heuristic_model):
    if not start or not goal:
        return []
    w, h = factory_map.width, factory_map.height

    def h_func(node):
        # node.x, node.y, goal.x, goal.y
        st = np.array([[node[0], node[1], goal[0], goal[1]]], dtype=np.float32)
        return estimate_cost(heuristic_model, st)

    directions = ACTIONS_8DIR
    open_set = []
    heapq.heappush(open_set, (h_func(start), start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        cur_f, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in directions:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0<=nx<w and 0<=ny<h:
                if not factory_map.is_obstacle(nx, ny):
                    tentative_g = g_score[cur]+1
                    if (nx, ny) not in g_score or tentative_g<g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        came_from[(nx, ny)] = cur
                        f_score = tentative_g + h_func((nx, ny))
                        heapq.heappush(open_set, (f_score,(nx, ny)))
    return []

# Placeholder Deep Q
def compute_route_deepq_8dir(factory_map, start, goal, train_model=False):
    # Ca demo, facem BFS
    return compute_route_bfs_8dir(factory_map, start, goal)

##############################################################################
# CLASE AUXILIARE (FactoryMap, WarehouseData etc.)
##############################################################################
class FactoryMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[False]*width for _ in range(height)]
        self.start_points = [None, None, None]
        self.warehouse_zones = [None, None, None, None]

    def toggle_obstacle(self, x, y):
        if 0<=x<self.width and 0<=y<self.height:
            self.grid[y][x] = not self.grid[y][x]

    def is_obstacle(self, x, y):
        if 0<=x<self.width and 0<=y<self.height:
            return self.grid[y][x]
        return True

    def set_start(self, index, x, y):
        if 0<=x<self.width and 0<=y<self.height:
            self.start_points[index] = (x, y)

    def set_warehouse_zone(self, index, x, y):
        if 0<=x<self.width and 0<=y<self.height:
            self.warehouse_zones[index] = (x, y)


class WarehouseData:
    def __init__(self):
        self.data = [
            {"capacity":10,"used":0,"component":"Mix1"},
            {"capacity":10,"used":0,"component":"Mix2"},
            {"capacity":10,"used":0,"component":"Mix3"},
            {"capacity":10,"used":0,"component":"Mix4"},
        ]
    def is_full(self,idx):
        d = self.data[idx]
        return d["used"]>=d["capacity"]
    def add_one_box(self,idx):
        d = self.data[idx]
        if d["used"]<d["capacity"]:
            d["used"]+=1
    def get_info_as_string(self,idx):
        d = self.data[idx]
        return f"Capacitate:{d['capacity']} Used:{d['used']} Comp:{d['component']}"

class AssignDestination:
    def __init__(self, wh_data):
        self.wh_data = wh_data
    def assign_part_to_destination(self, w_index):
        if w_index<len(self.wh_data.data) and not self.wh_data.is_full(w_index):
            return w_index
        if len(self.wh_data.data)>3 and not self.wh_data.is_full(3):
            return 3
        return None

class RobotStateIndicator:
    RED="red"
    YELLOW="yellow"
    GREEN="green"
    def __init__(self, led_canvas):
        self.current_state=self.RED
        self.led_canvas=led_canvas
        self.draw_led()

    def set_state(self, state):
        self.current_state=state
        self.draw_led()

    def draw_led(self):
        self.led_canvas.delete("all")
        self.led_canvas.create_oval(10,10,40,40,fill=self.current_state)

##############################################################################
# Barcode
##############################################################################
def build_barcode_model():
    model=keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_barcode_model(model):
    X=np.random.rand(1000,100)
    y=np.random.randint(0,2,(1000,))
    model.fit(X,y,epochs=3,verbose=0)
    return model

barcode_model=build_barcode_model()
barcode_model=train_barcode_model(barcode_model)

def detect_barcode_with_nn(frame,threshold=0.8):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(10,10))
    inp=resized.flatten().astype('float32')/255.0
    inp=inp.reshape(1,100)
    pred=barcode_model.predict(inp,verbose=0)[0]
    cls=np.argmax(pred)
    confidence=pred[cls]
    if confidence<threshold:
        return None
    else:
        return "2-1-309" if cls==0 else "3-4-101"

def detect_barcode(image):
    pil_img=PILImage.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    barcodes=decode(pil_img)
    if barcodes:
        bc=barcodes[0]
        return bc.data.decode('utf-8')
    return None

##############################################################################
# VisionSystem
##############################################################################
class VisionSystem:
    def __init__(self, part_db):
        self.part_database=part_db
    def get_part_info(self, code_data):
        if "309" in code_data:
            return ("309","ProjB",5)
        if "101" in code_data:
            return ("101","ProjA",10)
        return (None,None,None)

##############################################################################
# LOGICA ROBOTULUI
##############################################################################
def compute_route_deepq_8dir(factory_map, start_coord, wh_coord, train_model=True):
    # placeholder
    return compute_route_bfs_8dir(factory_map, start_coord, wh_coord)

class RobotSimulator:
    def __init__(self, app, led_indicator, factory_map, wh_data, vision):
        self.app=app
        self.led_indicator=led_indicator
        self.factory_map=factory_map
        self.wh_data=wh_data
        self.vision=vision
        self.delivery_in_progress=False
        self.current_task=None
        self.start_index=0
        self.warehouse_index=0
        self.final_warehouse_index=None

        self.route_forward=[]
        self.route_forward_index=0
        self.route_return=[]
        self.route_return_index=0
        self.robot_position=None
        self.PLAN_METHOD="bfs"

    def process_part(self, code_data):
        if self.delivery_in_progress:
            messagebox.showinfo("Info","Există deja o livrare în curs!")
            return
        self.show_barcode_details(code_data)
        parts=code_data.split("-")
        if len(parts)>=2:
            try:
                self.start_index=int(parts[0])-1
                self.warehouse_index=int(parts[1])-1
            except:
                self.start_index=0
                self.warehouse_index=0

        pid,proj,req=self.vision.get_part_info(code_data)
        self.current_task=(pid,proj)
        if self.factory_map.start_points[self.start_index] is None:
            messagebox.showwarning("Verificare",f"Start #{self.start_index+1} nu e setat!")
            return
        if self.warehouse_index<0 or self.warehouse_index>=len(self.factory_map.warehouse_zones):
            messagebox.showwarning("Verificare",f"Warehouse #{self.warehouse_index+1} nu există!")
            return
        if self.factory_map.warehouse_zones[self.warehouse_index] is None:
            messagebox.showwarning("Verificare",f"Warehouse #{self.warehouse_index+1} nu e setat!")
            return

        self.app.close_camera()
        self.delivery_in_progress=True
        self.led_indicator.set_state(RobotStateIndicator.YELLOW)
        self.led_indicator.led_canvas.after(1000,self.start_transport)

    def show_barcode_details(self, code_data):
        parts=code_data.split("-")
        s_str=parts[0] if parts else "?"
        w_str=parts[1] if len(parts)>1 else "?"
        pid,proj,req=self.vision.get_part_info(code_data)
        details_text=(f"Cod detectat:{code_data}\n"
                      f"Start indicat:{s_str}\n"
                      f"Warehouse indicat:{w_str}\n"
                      f"Piesa:{pid if pid else 'necunoscut'}\n"
                      f"Proiect:{proj if proj else 'necunoscut'}\n"
                      f"Cantitate necesara:{req if req else 'N/A'}")
        self.app.details_label.config(text=details_text)

    def start_transport(self):
        if not self.current_task:
            return
        wh_idx=self.app.assign_dest.assign_part_to_destination(self.warehouse_index)
        if wh_idx is None:
            messagebox.showwarning("Info","Depozitul cerut + fallback sunt pline!\nNu putem face livrarea.")
            self.delivery_in_progress=False
            return
        self.final_warehouse_index=wh_idx
        start_coord=self.factory_map.start_points[self.start_index]
        wh_coord=self.factory_map.warehouse_zones[wh_idx]

        if self.PLAN_METHOD=="bfs":
            path_fwd=compute_route_bfs_8dir(self.factory_map, start_coord, wh_coord)
        elif self.PLAN_METHOD=="q_table":
            messagebox.showinfo("Q-learning","Antrenare Q-learning tabular.")
            path_fwd=compute_route_qlearning_8dir(self.factory_map, start_coord, wh_coord)
        elif self.PLAN_METHOD=="neural_dstar":
            messagebox.showinfo("Neural D*","Calcul Neural D*.")
            model_heur=build_or_load_heuristic_model("heuristic_model.keras",False)
            path_fwd=compute_route_neural_dstar(self.factory_map, start_coord, wh_coord, model_heur)
        else:
            messagebox.showinfo("Deep Q","Placeholder Deep Q.")
            path_fwd=compute_route_deepq_8dir(self.factory_map, start_coord, wh_coord,train_model=False)

        self.route_forward=path_fwd
        self.route_forward_index=0
        if len(self.route_forward)==0:
            messagebox.showwarning("Verificare","Nu există rută la warehouse!")
            self.delivery_in_progress=False
            return
        self.led_indicator.set_state(RobotStateIndicator.GREEN)
        self.move_forward_step()

    def move_forward_step(self):
        if self.route_forward_index<len(self.route_forward):
            nx,ny=self.route_forward[self.route_forward_index]
            if self.factory_map.is_obstacle(nx,ny):
                cur_pos=self.robot_position or self.factory_map.start_points[self.start_index]
                wh_coord=self.factory_map.warehouse_zones[self.final_warehouse_index]
                if self.PLAN_METHOD=="bfs":
                    new_route=compute_route_bfs_8dir(self.factory_map, cur_pos, wh_coord)
                elif self.PLAN_METHOD=="q_table":
                    new_route=compute_route_qlearning_8dir(self.factory_map, cur_pos, wh_coord)
                elif self.PLAN_METHOD=="neural_dstar":
                    model_heur=build_or_load_heuristic_model("heuristic_model.keras",False)
                    new_route=compute_route_neural_dstar(self.factory_map, cur_pos, wh_coord, model_heur)
                else:
                    new_route=compute_route_deepq_8dir(self.factory_map, cur_pos, wh_coord,False)
                if not new_route:
                    messagebox.showwarning("Eroare","Nu mai există drum spre depozit (obstacol)!")
                    self.reset_robot()
                    return
                else:
                    self.route_forward=new_route
                    self.route_forward_index=0
                    nx,ny=self.route_forward[0]
            self.robot_position=(nx,ny)
            self.route_forward_index+=1
            self.app.draw_factory_map()
            self.led_indicator.led_canvas.after(300,self.move_forward_step)
        else:
            self.finish_delivery()

    def finish_delivery(self):
        self.led_indicator.set_state(RobotStateIndicator.RED)
        self.ask_delivery_confirmation()

    def ask_delivery_confirmation(self):
        top=tk.Toplevel(self.app.root)
        top.title("Confirmare Livrare")
        top.lift()
        top.attributes("-topmost",True)
        top.after_idle(top.attributes,'-topmost',False)
        lbl=tk.Label(top,text="A fost livrat corect?")
        lbl.pack(padx=10,pady=10)
        frm=tk.Frame(top)
        frm.pack(pady=5)
        btn_yes=tk.Button(frm,text="Da",command=lambda:self.delivery_confirmed(top))
        btn_yes.pack(side="left",padx=5)
        btn_no=tk.Button(frm,text="Nu",command=lambda:self.delivery_failed(top))
        btn_no.pack(side="left",padx=5)

    def delivery_confirmed(self, popup):
        popup.destroy()
        self.app.wh_data.add_one_box(self.final_warehouse_index)
        messagebox.showinfo("Info","Livrare confirmată. Robotul se întoarce la start.")
        self.return_to_band()

    def delivery_failed(self, popup):
        popup.destroy()
        messagebox.showwarning("Info","Livrarea eșuată. Resetăm robotul.")
        self.reset_robot()

    def return_to_band(self):
        wh_coord=self.factory_map.warehouse_zones[self.final_warehouse_index]
        st_coord=self.factory_map.start_points[self.start_index]
        if self.PLAN_METHOD=="bfs":
            path_ret=compute_route_bfs_8dir(self.factory_map, wh_coord, st_coord)
        elif self.PLAN_METHOD=="q_table":
            path_ret=compute_route_qlearning_8dir(self.factory_map, wh_coord, st_coord)
        elif self.PLAN_METHOD=="neural_dstar":
            model_heur=build_or_load_heuristic_model("heuristic_model.keras",False)
            path_ret=compute_route_neural_dstar(self.factory_map, wh_coord, st_coord, model_heur)
        else:
            path_ret=compute_route_deepq_8dir(self.factory_map, wh_coord, st_coord,False)

        self.route_return=path_ret
        self.route_return_index=0
        if len(self.route_return)==0:
            messagebox.showwarning("Verificare","Nu există rută de întoarcere!")
            self.reset_robot()
            return
        self.led_indicator.set_state(RobotStateIndicator.GREEN)
        self.move_return_step()

    def move_return_step(self):
        if self.route_return_index<len(self.route_return):
            nx,ny=self.route_return[self.route_return_index]
            if self.factory_map.is_obstacle(nx,ny):
                cur_pos=self.robot_position or self.factory_map.warehouse_zones[self.final_warehouse_index]
                st_coord=self.factory_map.start_points[self.start_index]
                if self.PLAN_METHOD=="bfs":
                    new_route=compute_route_bfs_8dir(self.factory_map, cur_pos, st_coord)
                elif self.PLAN_METHOD=="q_table":
                    new_route=compute_route_qlearning_8dir(self.factory_map, cur_pos, st_coord)
                elif self.PLAN_METHOD=="neural_dstar":
                    model_heur=build_or_load_heuristic_model("heuristic_model.keras",False)
                    new_route=compute_route_neural_dstar(self.factory_map, cur_pos, st_coord, model_heur)
                else:
                    new_route=compute_route_deepq_8dir(self.factory_map, cur_pos, st_coord,False)
                if not new_route:
                    messagebox.showwarning("Eroare","Nu mai există drum de întoarcere!")
                    self.reset_robot()
                    return
                else:
                    self.route_return=new_route
                    self.route_return_index=0
                    nx,ny=self.route_return[0]

            self.robot_position=(nx,ny)
            self.route_return_index+=1
            self.app.draw_factory_map()
            self.led_indicator.led_canvas.after(300,self.move_return_step)
        else:
            messagebox.showinfo("Info","Robotul s-a întors la start.")

            sx,sy=self.factory_map.start_points[self.start_index]
            wx,wy=self.factory_map.warehouse_zones[self.final_warehouse_index]
            webbrowser.open(
                f"http://127.0.0.1:5000/compare-route?"
                f"method2={self.PLAN_METHOD}&"
                f"start_x={sx}&start_y={sy}&end_x={wx}&end_y={wy}"
            )
            self.reset_robot()

    def reset_robot(self):
        self.led_indicator.set_state(RobotStateIndicator.RED)
        self.delivery_in_progress=False
        self.current_task=None
        self.route_forward=[]
        self.route_forward_index=0
        self.route_return=[]
        self.route_return_index=0
        self.robot_position=None
        self.final_warehouse_index=None
        self.app.draw_factory_map()
        self.app.start_camera_capture()

##############################################################################
# APLICAȚIA TKINTER
##############################################################################
class Application:
    def __init__(self, root):
        self.root=root
        self.root.title("Demo BFS+Q-table+DeepQ+NeuralD*+Barcode")

        self.factory_map=FactoryMap(20,10)
        self.cell_size=20
        self.wh_data=WarehouseData()
        self.assign_dest=AssignDestination(self.wh_data)
        part_db={"P1":{"project":"ProjA","required_qty":10},
                 "P2":{"project":"ProjB","required_qty":5}}
        self.vision=VisionSystem(part_db)

        frame_left=tk.Frame(root)
        frame_left.pack(side="left",padx=10,pady=10)

        frame_right=tk.Frame(root)
        frame_right.pack(side="right",padx=10,pady=10)

        self.factory_canvas=tk.Canvas(frame_left,
                                      width=self.factory_map.width*self.cell_size,
                                      height=self.factory_map.height*self.cell_size,
                                      bg="white")
        self.factory_canvas.pack()
        self.factory_canvas.bind("<Button-1>",self.on_map_click)
        self.factory_canvas.bind("<Motion>",self.on_map_mouse_move)

        self.map_mode=tk.StringVar(value="obstacle")
        fmodes=tk.Frame(frame_left)
        fmodes.pack(pady=5)
        tk.Radiobutton(fmodes,text="Obstacle",variable=self.map_mode,value="obstacle").pack(side="left")
        tk.Radiobutton(fmodes,text="Start #1",variable=self.map_mode,value="start1").pack(side="left")
        tk.Radiobutton(fmodes,text="Start #2",variable=self.map_mode,value="start2").pack(side="left")
        tk.Radiobutton(fmodes,text="Start #3",variable=self.map_mode,value="start3").pack(side="left")

        fmodes2=tk.Frame(frame_left)
        fmodes2.pack(pady=5)
        tk.Radiobutton(fmodes2,text="WH #1",variable=self.map_mode,value="warehouse1").pack(side="left")
        tk.Radiobutton(fmodes2,text="WH #2",variable=self.map_mode,value="warehouse2").pack(side="left")
        tk.Radiobutton(fmodes2,text="WH #3",variable=self.map_mode,value="warehouse3").pack(side="left")
        tk.Radiobutton(fmodes2,text="WH #4",variable=self.map_mode,value="warehouse4").pack(side="left")

        flayout=tk.Frame(frame_left)
        flayout.pack(pady=5)
        tk.Button(flayout,text="Save Layout",command=self.save_layout).pack(side="left",padx=5)
        tk.Button(flayout,text="Load Layout",command=self.load_layout).pack(side="left",padx=5)
        tk.Button(flayout,text="Reset Layout",command=self.reset_layout).pack(side="left",padx=5)

        fplan=tk.LabelFrame(frame_left,text="Plan Method")
        fplan.pack(pady=5)
        self.plan_method_var=tk.StringVar(value="bfs")
        tk.Radiobutton(fplan,text="BFS",variable=self.plan_method_var,value="bfs").pack(anchor="w")
        tk.Radiobutton(fplan,text="Q-table",variable=self.plan_method_var,value="q_table").pack(anchor="w")
        tk.Radiobutton(fplan,text="Deep Q",variable=self.plan_method_var,value="deep_q").pack(anchor="w")
        tk.Radiobutton(fplan,text="Neural D*",variable=self.plan_method_var,value="neural_dstar").pack(anchor="w")

        fled=tk.LabelFrame(frame_right,text="Robot State")
        fled.pack(pady=5)
        led_canvas=tk.Canvas(fled,width=50,height=50)
        led_canvas.pack()
        self.robot_state=RobotStateIndicator(led_canvas)
        self.robot=RobotSimulator(self,self.robot_state,self.factory_map,self.wh_data,self.vision)

        self.info_label=tk.Label(frame_right,text="Alege BFS/Q-table/Deep Q/Neural D*. Apoi scanează sau FakeBarcode.")
        self.info_label.pack(pady=10)
        self.details_label=tk.Label(frame_right,text="Nimic scanat încă.")
        self.details_label.pack(pady=10)
        tk.Button(frame_right,text="Fake Barcode",command=self.fake_barcode).pack(pady=5)
        tk.Button(frame_right,text="Start Camera",command=self.start_camera_capture).pack(pady=5)
        self.hover_label=tk.Label(frame_right,text="",fg="blue")
        self.hover_label.pack(pady=5)

        self.stop_camera_flag=False
        self.camera_thread=None
        self.draw_factory_map()

        # Global map pt Flask
        global GLOBAL_FACTORY_MAP
        GLOBAL_FACTORY_MAP=self.factory_map

        webbrowser.open("http://127.0.0.1:5000/")

    def draw_factory_map(self):
        self.factory_canvas.delete("all")
        for y in range(self.factory_map.height):
            for x in range(self.factory_map.width):
                color="black" if self.factory_map.is_obstacle(x,y) else "white"
                self.factory_canvas.create_rectangle(
                    x*self.cell_size, y*self.cell_size,
                    (x+1)*self.cell_size, (y+1)*self.cell_size,
                    fill=color, outline="gray"
                )
        start_colors=["green","darkgreen","lightgreen"]
        for i,sp in enumerate(self.factory_map.start_points):
            if sp:
                sx,sy=sp
                self.factory_canvas.create_rectangle(
                    sx*self.cell_size, sy*self.cell_size,
                    (sx+1)*self.cell_size, (sy+1)*self.cell_size,
                    fill=start_colors[i],outline="gray"
                )
        wh_colors=["blue","cyan","dodgerblue","navy"]
        for i,wz in enumerate(self.factory_map.warehouse_zones):
            if wz:
                wx,wy=wz
                c=wh_colors[i] if i<len(wh_colors) else "blue"
                self.factory_canvas.create_rectangle(
                    wx*self.cell_size,wy*self.cell_size,
                    (wx+1)*self.cell_size,(wy+1)*self.cell_size,
                    fill=c,outline="gray"
                )
        # rute de dus
        route_fwd=self.robot.route_forward
        idx_fwd=self.robot.route_forward_index
        for ii in range(idx_fwd):
            rx,ry=route_fwd[ii]
            self.factory_canvas.create_rectangle(
                rx*self.cell_size,ry*self.cell_size,
                (rx+1)*self.cell_size,(ry+1)*self.cell_size,
                fill="orange",outline="gray"
            )
        # rute de întors
        route_ret=self.robot.route_return
        idx_ret=self.robot.route_return_index
        for ii in range(idx_ret):
            rx,ry=route_ret[ii]
            self.factory_canvas.create_rectangle(
                rx*self.cell_size,ry*self.cell_size,
                (rx+1)*self.cell_size,(ry+1)*self.cell_size,
                fill="blue",outline="gray"
            )
        # poziție robot
        if self.robot.robot_position:
            rx,ry=self.robot.robot_position
            self.factory_canvas.create_oval(
                rx*self.cell_size+5, ry*self.cell_size+5,
                rx*self.cell_size+self.cell_size-5,ry*self.cell_size+self.cell_size-5,
                fill="yellow",outline="red",width=2
            )

    def on_map_click(self, e):
        x=e.x//self.cell_size
        y=e.y//self.cell_size
        mode=self.map_mode.get()
        if mode=="obstacle":
            self.factory_map.toggle_obstacle(x,y)
        elif mode.startswith("start"):
            idx=int(mode[-1])-1
            self.factory_map.set_start(idx,x,y)
        elif mode.startswith("warehouse"):
            idx=int(mode[-1])-1
            self.factory_map.set_warehouse_zone(idx,x,y)
        self.draw_factory_map()

    def on_map_mouse_move(self, e):
        x=e.x//self.cell_size
        y=e.y//self.cell_size
        found_wh_idx=None
        for i,wz in enumerate(self.factory_map.warehouse_zones):
            if wz==(x,y):
                found_wh_idx=i
                break
        if found_wh_idx is not None:
            info_str=self.wh_data.get_info_as_string(found_wh_idx)
            self.hover_label.config(text=f"WH #{found_wh_idx+1} => {info_str}")
        else:
            self.hover_label.config(text="")

    def fake_barcode(self):
        code_data="2-1-309"
        self.robot.PLAN_METHOD=self.plan_method_var.get()
        self.robot.process_part(code_data)

    def save_layout(self):
        data={
            "width":self.factory_map.width,
            "height":self.factory_map.height,
            "obstacles":[]
        }
        for y in range(self.factory_map.height):
            row=[]
            for x in range(self.factory_map.width):
                row.append(1 if self.factory_map.grid[y][x] else 0)
            data["obstacles"].append(row)
        try:
            with open("layout.json","w") as f:
                json.dump(data,f)
            messagebox.showinfo("Info","Layout salvat in layout.json")
        except Exception as e:
            messagebox.showerror("Eroare",f"Nu am reusit sa salvez {e}")

    def load_layout(self):
        try:
            with open("layout.json","r") as f:
                data=json.load(f)
            w=data["width"]
            h=data["height"]
            obs=data["obstacles"]
            for yy in range(min(h,self.factory_map.height)):
                for xx in range(min(w,self.factory_map.width)):
                    self.factory_map.grid[yy][xx]=(obs[yy][xx]==1)
            self.draw_factory_map()
            messagebox.showinfo("Info","Layout incarcat")
        except FileNotFoundError:
            messagebox.showerror("Eroare","Nu am gasit layout.json!")
        except Exception as e:
            messagebox.showerror("Eroare",f"Nu pot citi layout.json: {e}")

    def reset_layout(self):
        for y in range(self.factory_map.height):
            for x in range(self.factory_map.width):
                self.factory_map.grid[y][x]=False
        self.draw_factory_map()
        messagebox.showinfo("Info","Layout resetat.")

    def start_camera_capture(self):
        if self.camera_thread and self.camera_thread.is_alive():
            messagebox.showinfo("Info","Camera already running!")
            return
        self.stop_camera_flag=False
        self.camera_thread=threading.Thread(target=self.camera_loop)
        self.camera_thread.start()

    def close_camera(self):
        self.stop_camera_flag=True

    def camera_loop(self):
        cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        last_scanned=None
        while not self.stop_camera_flag:
            ret,frame=cap.read()
            if not ret:
                break
            code_data=detect_barcode(frame)
            if code_data and code_data!=last_scanned:
                last_scanned=code_data
                self.robot.PLAN_METHOD=self.plan_method_var.get()
                self.root.after(0,lambda c=code_data: self.barcode_detected(c))
            cv2.imshow("Camera Preview - Press q to quit",frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.camera_thread=None

    def barcode_detected(self, code_data):
        self.robot.process_part(code_data)

    def on_close(self):
        self.stop_camera_flag=True
        self.root.destroy()

def main():
    root=tk.Tk()
    app=Application(root)
    root.protocol("WM_DELETE_WINDOW",app.on_close)
    root.mainloop()

##############################################################################
# LOGICA WEB (FLASK)
##############################################################################
from flask import Flask, request, render_template_string, send_file

app_flask=Flask(__name__)
GLOBAL_FACTORY_MAP=None

# 1) Combinăm traseul de dus cu traseul de întors într-o singură listă
def combine_routes(route_fwd, route_ret):
    if not route_fwd:
        return route_ret
    if not route_ret:
        return route_fwd
    return route_fwd + route_ret[1:]  # evităm duplicarea nodului comun

# 2) compute_cost_reward - îl definim cu recompense diferite la BFS vs Q_table vs altele
def compute_cost_reward(route, method="bfs"):
    if not route:
        return None,None
    cost=list(range(len(route)))  # cost = index de pas
    if method=="bfs":
        # ex: 0 per pas, +1 la final
        rewards=[0]*(len(route)-1)+[1.0]
    elif method=="q_table":
        # penalizare -0.1, +10 la final
        rewards=[-0.1]*(len(route)-1)+[10.0]
    elif method=="neural_dstar":
        # penalizare -0.05, +5 la final
        rewards=[-0.05]*(len(route)-1)+[5.0]
    elif method=="deep_q":
        # penalizare -0.15, +10 la final
        rewards=[-0.15]*(len(route)-1)+[10.0]
    else:
        rewards=[0]*(len(route)-1)+[1.0]

    cum_rew=np.cumsum(rewards)
    return cost, cum_rew

def plot_cost_reward(cost, rewards):
    fig, ax=plt.subplots(figsize=(4,3), dpi=100)
    ax.plot(cost, marker='o', label='Cost (pas)', color='blue')
    ax.plot(rewards, marker='o', label='Cumulative Reward', color='green')
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig

# 3) desenăm hartă + două segmente (dus=orange, întors=blue) + legend
def draw_map_and_path_two_segments(factory_map, route_fwd, route_ret):
    grid=np.array(factory_map.grid, dtype=np.float32)
    fig, ax=plt.subplots(figsize=(factory_map.width/2, factory_map.height/2), dpi=100)
    ax.imshow(grid, cmap='gray_r', origin='upper')

    # dus = portocaliu
    if route_fwd and len(route_fwd)>1:
        xs=[p[0] for p in route_fwd]
        ys=[p[1] for p in route_fwd]
        ax.plot(xs, ys, marker='o', color='orange', linewidth=2, label="Dus")

    # întors = albastru
    if route_ret and len(route_ret)>1:
        xs=[p[0] for p in route_ret]
        ys=[p[1] for p in route_ret]
        ax.plot(xs, ys, marker='o', color='blue', linewidth=2, label="Întors")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Map (Dus+Întors)")
    ax.legend()
    fig.tight_layout()
    return fig

def create_composite_with_return(factory_map, route_fwd, route_ret, method_name):
    # harta: segmente diferite
    fig_map=draw_map_and_path_two_segments(factory_map, route_fwd, route_ret)

    # combinăm rutele pt a calcula cost/reward pe TOT parcursul
    full_route=combine_routes(route_fwd, route_ret)
    cost, rew=compute_cost_reward(full_route, method_name)
    if cost is None or rew is None:
        fig_graph, ax=plt.subplots(figsize=(4,3), dpi=100)
        ax.text(0.5,0.5, "No Data", ha='center', va='center', color='red')
        ax.axis('off')
    else:
        fig_graph=plot_cost_reward(cost, rew)

    # Creăm imagine finală (două subplots: sus harta, jos cost-reward)
    composite_fig, (ax1, ax2)=plt.subplots(2,1, figsize=(max(factory_map.width/2,4), factory_map.height/2+3), dpi=100)
    # Redesenăm harta:
    grid=np.array(factory_map.grid, dtype=np.float32)
    ax1.imshow(grid, cmap='gray_r', origin='upper')
    if route_fwd and len(route_fwd)>1:
        xsf=[p[0] for p in route_fwd]
        ysf=[p[1] for p in route_fwd]
        ax1.plot(xsf, ysf, marker='o', color='orange', linewidth=2, label="Dus")
    if route_ret and len(route_ret)>1:
        xsr=[p[0] for p in route_ret]
        ysr=[p[1] for p in route_ret]
        ax1.plot(xsr, ysr, marker='o', color='blue', linewidth=2, label="Întors")
    ax1.set_title(f"Method: {method_name} - Map (Dus + Întors)")
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Grafic cost & reward
    if cost is None or rew is None:
        ax2.text(0.5,0.5,"No Data",ha='center',va='center',color='red')
        ax2.axis('off')
    else:
        ax2.plot(cost, marker='o', label="Cost (pas)", color='blue')
        ax2.plot(rew, marker='o', label="Cumulative Reward", color='green')
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Value")
        ax2.legend()

    composite_fig.tight_layout()
    return composite_fig


@app_flask.route("/")
def index_page():
    return """<h1>Flask-based Route Demo</h1>
              <p>Endpoint: /compare-route?method2=q_table&start_x=0&start_y=0&end_x=9&end_y=9<br/>
              Vei vedea BFS (stânga) vs method2 (dreapta), inclusiv dus + întors, cost/pasi, reward cumulativ.<br/>
              Legenda culori pe hartă: portocaliu = dus, albastru = întors.<br/>
              Legenda curbe jos: albastru = cost (pas), verde = cumulative reward.
              </p>"""


@app_flask.route("/compare-route")
def compare_route():
    global GLOBAL_FACTORY_MAP
    if GLOBAL_FACTORY_MAP is None:
        return "Nu există încă FactoryMap activ",404

    method2=request.args.get("method2","bfs")
    sx=int(request.args.get("start_x",0))
    sy=int(request.args.get("start_y",0))
    ex=int(request.args.get("end_x",0))
    ey=int(request.args.get("end_y",0))

    # BFS forward/back
    route_fwd_bfs=compute_route_bfs_8dir(GLOBAL_FACTORY_MAP,(sx,sy),(ex,ey))
    route_ret_bfs=compute_route_bfs_8dir(GLOBAL_FACTORY_MAP,(ex,ey),(sx,sy))
    comp1=create_composite_with_return(GLOBAL_FACTORY_MAP,route_fwd_bfs, route_ret_bfs, "bfs")

    # method2 forward/back
    if method2=="bfs":
        route_fwd_2=route_fwd_bfs
        route_ret_2=route_ret_bfs
    elif method2=="q_table":
        route_fwd_2=compute_route_qlearning_8dir(GLOBAL_FACTORY_MAP,(sx,sy),(ex,ey))
        route_ret_2=compute_route_qlearning_8dir(GLOBAL_FACTORY_MAP,(ex,ey),(sx,sy))
    elif method2=="neural_dstar":
        model_heur=build_or_load_heuristic_model("heuristic_model.keras",False)
        route_fwd_2=compute_route_neural_dstar(GLOBAL_FACTORY_MAP,(sx,sy),(ex,ey),model_heur)
        route_ret_2=compute_route_neural_dstar(GLOBAL_FACTORY_MAP,(ex,ey),(sx,sy),model_heur)
    elif method2=="deep_q":
        route_fwd_2=compute_route_deepq_8dir(GLOBAL_FACTORY_MAP,(sx,sy),(ex,ey),False)
        route_ret_2=compute_route_deepq_8dir(GLOBAL_FACTORY_MAP,(ex,ey),(sx,sy),False)
    else:
        route_fwd_2=route_fwd_bfs
        route_ret_2=route_ret_bfs

    comp2=create_composite_with_return(GLOBAL_FACTORY_MAP,route_fwd_2, route_ret_2, method2)

    # Lipim cele 2 imagini orizontal
    buf1=BytesIO()
    comp1.savefig(buf1, format='PNG')
    plt.close(comp1)
    buf1.seek(0)

    buf2=BytesIO()
    comp2.savefig(buf2, format='PNG')
    plt.close(comp2)
    buf2.seek(0)

    img1=PILImage.open(buf1)
    img2=PILImage.open(buf2)
    margin=20
    total_width=img1.width+img2.width+margin
    max_h=max(img1.height,img2.height)
    final_image=PILImage.new("RGB",(total_width,max_h), color="black")
    final_image.paste(img1,(0,0))
    final_image.paste(img2,(img1.width+margin,0))
    final_buf=BytesIO()
    final_image.save(final_buf,format='PNG')
    final_buf.seek(0)
    return send_file(final_buf, mimetype="image/png")

def start_flask_server():
    app_flask.run(debug=True, port=5000, use_reloader=False)

if __name__=="__main__":
    flask_thread=threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    webbrowser.open("http://127.0.0.1:5000/")
    main()
