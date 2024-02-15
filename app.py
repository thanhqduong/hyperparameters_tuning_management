import random
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import sqlite3

from model import Model
db_file = 'database/main.db'
con = sqlite3.connect(db_file,  check_same_thread=False)
db_lock = threading.Lock()


async_mode = None

app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode)



def background_task():
    while True:
        with db_lock:
            cur = con.cursor()
            sql_get_pending = 'SELECT learning_rate, dropout, batch_size, num_epochs, optimizer, order_id FROM pending ORDER BY order_id LIMIT 1;'
            res = cur.execute(sql_get_pending)
            r_list = res.fetchall()
            cur.close()
        if len(r_list) > 0:
            r = r_list[0]

            ret = dict()
            ret['learning_rate'] = r[0]
            ret['num_epochs'] = r[3]
            ret['batch_size'] = r[2]
            ret['dropout'] = r[1]
            ret['optimizer'] = r[4]

            train_result = Model(r[0], r[1], r[2], r[3], r[4]).train()
            ret['runtime'] = train_result[0]
            ret['accuracy'] = train_result[1]
            ret['f1_score'] = train_result[2]
            ret['losses'] = train_result[3]

            with db_lock:
                cur = con.cursor()
                sql_drop_row = f'DELETE FROM pending WHERE order_id = {r[5]};'
                res = cur.execute(sql_drop_row)
                con.commit()
                cur.close()
            add_result([ret])


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.event
def add_manual_search(message):
    to_add_popup([message['data']])



@socketio.event
def add_random_search(message):
    form_data = dict()
    opt = []
    for key in message['data']:
        if not key.startswith('optimizer'):
            form_data[key] = float(message['data'][key])
        else:
            opt.append(message['data'][key])
    data = []
    for _ in range(int(form_data['num_count'])):
        try:
            r = dict()
            r['learning_rate'] = round(random.uniform(form_data['min_learning_rate'], form_data['max_learning_rate']),4)
            r['num_epochs'] = round(random.uniform(form_data['min_num_epochs'], form_data['max_num_epochs']))
            r['batch_size'] = round(random.uniform(form_data['min_batch_size'], form_data['max_batch_size']))
            r['dropout'] = round(random.uniform(form_data['min_dropout'], form_data['max_dropout']),2)
            r['optimizer'] = random.choice(opt)
            data.append(r)
        except:
            pass
    to_add_popup(data)

def get_grid_list(min_val, max_val, step):
    ret = [min_val]
    while ret[-1] <= max_val and step >= 0.00001:
        val = int((ret[-1] + step) * 10000)/10000 # avoid computation error
        ret.append(val)
    return ret

@socketio.event
def add_grid_search(message):
    form_data = dict()
    opt = []
    for key in message['data']:
        if not key.startswith('optimizer'):
            form_data[key] = float(message['data'][key])
        else:
            opt.append(message['data'][key])
    data = []
    lr_list = get_grid_list(form_data['min_learning_rate'], form_data['max_learning_rate'], form_data['step_learning_rate'])
    ep_list = get_grid_list(form_data['min_num_epochs'], form_data['max_num_epochs'], form_data['step_num_epochs'])
    bs_list = get_grid_list(form_data['min_batch_size'], form_data['max_batch_size'], form_data['step_batch_size'])
    do_list = get_grid_list(form_data['min_dropout'], form_data['min_dropout'], form_data['step_dropout'])
    for lr in lr_list:
        for ep in ep_list:
            for bs in bs_list:
                for do in do_list:
                    for op in opt:
                        r = dict()
                        r['learning_rate'] = lr
                        r['num_epochs'] = ep
                        r['batch_size'] = bs
                        r['dropout'] = do
                        r['optimizer'] = op
                        data.append(r)

    to_add_popup(data)

def check_duplicate(data):
    with db_lock:
        cur = con.cursor()
        sql = 'SELECT num_epochs, learning_rate, dropout, batch_size, optimizer FROM result;'
        res = cur.execute(sql)
        already = res.fetchall()
        sql = 'SELECT num_epochs, learning_rate, dropout, batch_size, optimizer FROM pending;'
        res = cur.execute(sql)
        already.extend(res.fetchall())
        cur.close()
    already = [[str(a) for a in x] for x in already]
    to_search = []
    duplicate = []
    for d in data:
        dd = [int(d['num_epochs']), d['learning_rate'], d['dropout'], int(d['batch_size']), d['optimizer']]
        str_d = [str(element) for element in dd]
        if str_d in already:
            duplicate.append(dd)
        else:
            to_search.append(dd)
    return to_search, duplicate
        
def to_add_popup(data):
    to_search, duplicate = check_duplicate(data)
    data_1 = to_string_drag_table(to_search)
    data_2 = to_string_duplicate_table(duplicate)
    emit('popup_add', {'data_1': data_1, 'data_2': data_2})

def add_result(results):
    with db_lock:
        cur = con.cursor()
        for r in results:
            sql_insert_result = f"""INSERT INTO result (learning_rate,num_epochs,batch_size,dropout, optimizer, runtime, accuracy, f1_score, losses)
                                VALUES ( {r['learning_rate']},{r['num_epochs']},{r['batch_size']},{r['dropout']}, '{r['optimizer']}', {r['runtime']}, {r['accuracy']}, {r['f1_score']}, '{r['losses']}');"""
            res = cur.execute(sql_insert_result)
            con.commit()
        cur.close()
    # update_result()

def add_pending(results, drop_first):
    with db_lock:
        cur = con.cursor()
        if drop_first:
            sql_del = 'DELETE FROM pending;'
            res = cur.execute(sql_del)
            res.fetchall()
        sql_check_order = 'SELECT order_id FROM pending ORDER BY order_id DESC LIMIT 1;'
        res = cur.execute(sql_check_order)
        orders = res.fetchall()
        cur_count = 0
        if len(orders) > 0:
            cur_count = orders[0][0]
        for r in results:
            sql_insert_result = f"""INSERT INTO pending (order_id, learning_rate,num_epochs,batch_size,dropout, optimizer)
                                VALUES ({r['order_id'] + cur_count},{r['learning_rate']},{r['num_epochs']},{r['batch_size']},{r['dropout']}, '{r['optimizer']}');"""
            res = cur.execute(sql_insert_result)
        con.commit()
        cur.close()
    update_pending()

@socketio.event
def get_add_data(data): # add  to pending
    ret = []
    count = 1
    for d in data['data']:
        print(d)
        r = dict()
        r['order_id'] = count
        r['learning_rate'] = float(d[2])
        r['num_epochs'] = float(d[1])
        r['batch_size'] = float(d[4])
        r['dropout'] = float(d[3])
        r['optimizer'] = d[5]
        ret.append(r)
        count += 1
    add_pending(ret, False)

@socketio.event
def get_pending_data(data): # add  to pending
    ret = []
    count = 1
    for d in data['data']:
        r = dict()
        r['order_id'] = count
        r['learning_rate'] = float(d[1])
        r['num_epochs'] = float(d[2])
        r['batch_size'] = float(d[3])
        r['dropout'] = float(d[4])
        r['optimizer'] = d[5]
        ret.append(r)
        count += 1
    
    add_pending(ret, True)

def to_string_pending_row(data):
    d = data[0]
    row_str = f"""<tr draggable="true">
                        <td>running</td>
                        <td>{d[0]}</td>
                        <td>{d[1]}</td>
                        <td>{d[2]}</td>
                        <td>{d[3]}</td>
                        <td>{d[4]}</td>
                        <td></td>
                    </tr>"""

    return row_str

def to_string_drag_table(data):
    final_str = []
    for d in data:
        row_str = f"""<tr draggable="true">
                            <td class="handle">&#9776;</td>
                            <td>{d[0]}</td>
                            <td>{d[1]}</td>
                            <td>{d[2]}</td>
                            <td>{d[3]}</td>
                            <td>{d[4]}</td>
                            <td class="delete-button" onclick="deleteRow(this)">🗑️</td>
                        </tr>"""
        final_str.append(row_str)
    return '\n'.join(final_str)

def to_string_duplicate_table(data):
    final_str = []
    for d in data:
        row_str = f"""<tr>
                            <td>{d[0]}</td>
                            <td>{d[1]}</td>
                            <td>{d[2]}</td>
                            <td>{d[3]}</td>
                            <td>{d[4]}</td>
                        </tr>"""
        final_str.append(row_str)
    return '\n'.join(final_str)

def to_string_result_table(data):
    final_str = []
    for d in data:
        row_str = f"""<tr>
                            <td>{d[0]}</td>
                            <td>{d[1]}</td>
                            <td>{d[2]}</td>
                            <td>{d[3]}</td>
                            <td>{d[4]}</td>
                            <td>{d[5]}</td>
                            <td>{d[6]}</td>
                            <td>{d[7]}</td>
                        </tr>"""
        final_str.append(row_str)
    return '\n'.join(final_str)

def update_pending():
    with db_lock:
        cur = con.cursor()
        sql = 'SELECT num_epochs, learning_rate, dropout, batch_size, optimizer FROM pending ORDER BY order_id;'
        res = cur.execute(sql)
        data = res.fetchall()
        cur.close()
    try:
        inner_str = to_string_pending_row([data[0]]) + '\n' +  to_string_drag_table(data[1:])
    except:
        inner_str = ''
    emit('update_pending', {'data': inner_str})

def update_result():
    with db_lock:
        cur = con.cursor()
        sql = 'SELECT learning_rate, dropout, batch_size, num_epochs, optimizer, runtime, accuracy, f1_score FROM result;'
        res = cur.execute(sql)
        data = res.fetchall()
        cur.close()
    try:
        inner_str = to_string_result_table(data)
    except:
        inner_str = ''
    emit('update_result', {'data': inner_str})

@socketio.event
def connect():
    update_pending()
    update_result()
    background_thread = threading.Thread(target=background_task)
    background_thread.daemon = True
    background_thread.start()
    while True:
        time.sleep(200)
        update_result()
    


if __name__ == '__main__':
    socketio.run(app)
    
