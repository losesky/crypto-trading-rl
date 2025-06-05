// 检查model_comparison.html页面交互
// 在浏览器控制台中运行此脚本以测试页面交互

console.log("开始模拟测试...");

// 等待DOM和相关资源加载完成
setTimeout(() => {
  try {
    console.log("检查DOM元素...");
    
    // 检查关键元素
    const modelBtns = document.querySelectorAll('.model-btn');
    console.log(`找到 ${modelBtns.length} 个模型按钮`);
    
    const compareBtn = document.getElementById('compareBtn');
    console.log("比较按钮状态:", compareBtn ? "存在" : "不存在");
    
    const equityChart = document.getElementById('equityChart');
    console.log("权益图表状态:", equityChart ? "存在" : "不存在");
    
    const drawdownChart = document.getElementById('drawdownChart');
    console.log("回撤图表状态:", drawdownChart ? "存在" : "不存在");
    
    // 模拟点击第一个和第二个模型按钮
    console.log("模拟选择模型1和2...");
    if (modelBtns.length >= 2) {
      // 模型1应该已经激活
      modelBtns[1].click(); // 激活模型2
      
      // 检查激活状态
      console.log("模型1激活状态:", modelBtns[0].classList.contains('active'));
      console.log("模型2激活状态:", modelBtns[1].classList.contains('active'));
    }
    
    // 模拟点击比较按钮
    if (compareBtn) {
      console.log("模拟点击比较按钮...");
      compareBtn.click();
      
      // 检查按钮状态变化
      console.log("比较按钮文本:", compareBtn.innerText);
      console.log("比较按钮是否禁用:", compareBtn.disabled);
      
      // 检查body是否添加loading类
      console.log("Body是否有loading类:", document.body.classList.contains('loading'));
    }
    
    console.log("测试完成");
  } catch (error) {
    console.error("测试过程中出错:", error);
  }
}, 2000); // 等待2秒后执行测试
