// 一个简单的Node.js代理服务器，用于避免CORS问题
const http = require('http');
const https = require('https');
const url = require('url');

const PORT = 8097;  // 代理服务器端口
const BINANCE_API_BASE = 'https://testnet.binancefuture.com';

// 创建HTTP服务器
const server = http.createServer((req, res) => {
  // 设置CORS头
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type,X-MBX-APIKEY');
  res.setHeader('Access-Control-Allow-Credentials', true);

  // 处理OPTIONS预检请求
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // 解析请求URL
  const reqUrl = url.parse(req.url);
  const apiPath = reqUrl.pathname;
  const queryString = reqUrl.search || '';

  console.log(`代理请求: ${apiPath}${queryString}`);

  // 转发到币安API
  const options = {
    hostname: 'testnet.binancefuture.com',
    path: apiPath + queryString,
    method: req.method,
    headers: {
      'User-Agent': 'Mozilla/5.0',
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    }
  };

  // 如果有API密钥，添加到请求头
  if (req.headers['x-mbx-apikey']) {
    options.headers['X-MBX-APIKEY'] = req.headers['x-mbx-apikey'];
  }

  const proxyReq = https.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  // 处理错误
  proxyReq.on('error', (error) => {
    console.error(`代理请求错误:`, error);
    res.writeHead(500);
    res.end(JSON.stringify({ error: 'Proxy request failed' }));
  });

  // 如果是POST请求，转发请求体
  if (req.method === 'POST') {
    req.pipe(proxyReq, { end: true });
  } else {
    proxyReq.end();
  }
});

// 启动服务器
server.listen(PORT, () => {
  console.log(`代理服务器运行在 http://localhost:${PORT}`);
  console.log(`可以使用 http://localhost:${PORT}/fapi/v1/ticker/24hr 替代 ${BINANCE_API_BASE}/fapi/v1/ticker/24hr`);
});

console.log('启动代理服务器...');
