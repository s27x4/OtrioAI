let trainWS;
let gameWS;
const lossData = [];
let chart;

function connectTrainWS() {
  trainWS = new WebSocket(`ws://${location.host}/ws/train`);
  trainWS.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    document.getElementById('train_status').textContent = '学習中';
    document.getElementById('iteration').textContent = data.iteration;
    document.getElementById('loss').textContent = data.loss.toFixed(4);
    lossData.push({x: data.iteration, y: data.loss});
    chart.update();
  };
  trainWS.onclose = () => {
    document.getElementById('train_status').textContent = '停止';
  };
}

function connectGameWS() {
  gameWS = new WebSocket(`ws://${location.host}/ws/game`);
  gameWS.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    updateBoard(data.board);
    document.getElementById('winner').textContent = data.winner || '';
  };
}

function updateBoard(board) {
  const table = document.getElementById('board');
  table.innerHTML = '';
  for (let r = 0; r < 3; r++) {
    const tr = document.createElement('tr');
    for (let c = 0; c < 3; c++) {
      const td = document.createElement('td');
      let text = '';
      for (let s = 0; s < 3; s++) {
        const p = board[s][r][c];
        text += p === 0 ? '.' : p;
      }
      td.textContent = text;
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
}

async function post(url, obj) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(obj)
  });
  return res.json();
}

async function updateModelList() {
  const res = await fetch('/models');
  const data = await res.json();
  const select = document.getElementById('model_select');
  select.innerHTML = '';
  data.models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    select.appendChild(opt);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const ctx = document.getElementById('lossChart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {datasets: [{label: 'loss', data: lossData, fill: false, borderColor: 'blue'}]},
    options: {scales: {x: {type: 'linear', position: 'bottom'}}}
  });

  connectTrainWS();
  connectGameWS();
  updateModelList();

  document.getElementById('start_train').onclick = () => {
    const iterations = parseInt(document.getElementById('iterations').value || '1');
    post('/train', {iterations});
  };
  document.getElementById('stop_train').onclick = () => {
    post('/stop', {});
  };
  document.getElementById('save_model').onclick = () => {
    const name = document.getElementById('save_name').value || 'model.pt';
    post('/model_save', {path: name}).then(updateModelList);
  };
  document.getElementById('new_model').onclick = () => {
    post('/new_model', {}).then(updateModelList);
  };
  document.getElementById('load_model').onclick = () => {
    const name = document.getElementById('model_select').value;
    if (name) { post('/model_load', {path: name}); }
  };
  document.getElementById('start_game').onclick = () => {
    const name = document.getElementById('model_select').value;
    post('/start', {model: name});
  };
  document.getElementById('move_form').onsubmit = (e) => {
    e.preventDefault();
    const row = parseInt(document.getElementById('row').value);
    const col = parseInt(document.getElementById('col').value);
    const size = parseInt(document.getElementById('size').value);
    post('/move', {row, col, size}).then(updateBoard);
  };
});
