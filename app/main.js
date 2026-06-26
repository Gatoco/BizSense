const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'BizSense',
    backgroundColor: '#0f0f23',
    autoHideMenuBar: true,
    center: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

function startPythonServer() {
  const projectRoot = path.join(__dirname, '..');
  const venvPython = process.platform === 'win32'
    ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(projectRoot, '.venv', 'bin', 'python');

  const fs = require('fs');
  const logStream = fs.createWriteStream('/tmp/bizsense-backend.log', { flags: 'w' });

  pythonProcess = spawn(venvPython, ['-m', 'ml.main'], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    logStream.write(data);
  });

  pythonProcess.stderr.on('data', (data) => {
    logStream.write(data);
  });

  pythonProcess.on('close', (code) => {
    logStream.end();
  });
}

function waitForBackend(callback) {
  const maxAttempts = 15;
  let attempts = 0;

  const check = () => {
    const req = http.get('http://localhost:5000/docs', (res) => {
      if (res.statusCode === 200) {
        callback();
      } else {
        retry();
      }
    });

    req.on('error', () => {
      retry();
    });

    req.setTimeout(1000, () => {
      req.destroy();
      retry();
    });
  };

  const retry = () => {
    attempts++;
    if (attempts >= maxAttempts) {
      console.error('Backend no respondio despues de ' + maxAttempts + ' intentos');
      createWindow();
      return;
    }
    setTimeout(check, 500);
  };

  check();
}

app.whenReady().then(() => {
  startPythonServer();
  waitForBackend(() => {
    createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});