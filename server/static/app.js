// State Management
let singleFiles = [];
let batchFiles = [];
let currentIndex = { single: 0, batch: 0 };
let currentPage = 0;
let ITEMS_PER_PAGE = 20;
let trainChart = null;
let trainInterval = null;
let isPaused = false;
// ------------------------------
// 全域狀態與通訊
// ------------------------------
let activeDataset = JSON.parse(localStorage.getItem('ai_detector_active_ds') || 'null');

// 監聽來自 iframe 的消息 (跨 Port 通訊)
window.addEventListener('message', (event) => {
    const data = event.data;
    console.log("Shell Received Message:", data);

    if (data.type === 'DATASET_SELECTED') {
        activeDataset = data.dataset;
        localStorage.setItem('ai_detector_active_ds', JSON.stringify(activeDataset));
        
        // 轉發給標記模組 iframe (Port 8002)
        const labelingIframe = document.getElementById('iframe-labeling');
        if (labelingIframe && labelingIframe.contentWindow) {
            labelingIframe.contentWindow.postMessage({
                type: 'DATASET_SELECTED',
                dataset: activeDataset
            }, '*');
        }
        
        showAlert(`已切換工作區: ${activeDataset.name}`, "success");
    }
    
    if (data.type === 'REQUEST_DS') {
        // 子頁面請求當前資料集 (例如重新整理時)
        if (activeDataset) {
            event.source.postMessage({
                type: 'DATASET_SELECTED',
                dataset: activeDataset
            }, '*');
        }
    }
    
    if (data.type === 'SWITCH_PAGE') {
        switchPage(event.data.pageId);
    }
});

// 頁面加載時初始化
document.addEventListener('DOMContentLoaded', () => {
    updateSystemMonitor();
    setInterval(updateSystemMonitor, 5000);
    setInterval(checkAutoLabelStatus, 3000); // 每一秒檢查一次自動標記狀態
    
    // 恢復上次瀏覽的頁面
    const lastPage = localStorage.getItem('ai_detector_active_page') || 'datasets';
    switchPage(lastPage);

    checkPendingAgentInterrupt();
});

// 使用者在 confirm() 彈窗跳出來、還沒按下去之前關掉終端機／重新整理頁面，
// SSE 連線會直接斷掉、彈窗也隨之消失，但 SqliteSaver 其實還留著暫停在
// interrupt() 的 checkpoint（見 server/multi_agent.py 的 train_confirm_gate／
// save_offer_gate）。頁面重新載入時打 /api/agent/pending 用同一個 thread_id
// 查一次，卡著的話直接重新問一次、接著送 /api/agent/resume 繼續跑，不用
// 讓使用者自己發現「怎麼訓練確認就這樣消失了」。
async function checkPendingAgentInterrupt() {
    const threadId = sessionStorage.getItem(MANAGER_AGENT_THREAD_KEY);
    if (!threadId) return;

    const modelTypeEl = document.getElementById('manager-agent-engine-select');
    const modelType = modelTypeEl ? modelTypeEl.value : 'gemini';

    try {
        const res = await fetch(`/api/agent/pending?thread_id=${encodeURIComponent(threadId)}&model_type=${encodeURIComponent(modelType)}`);
        const data = await res.json();
        if (data.pending) {
            await finishOrResumeAgentTurn(
                { traceRefs: null, finalReply: null, sawError: false, interruptPayload: data.payload },
                threadId,
                modelType
            );
        }
    } catch (e) {
        console.error('檢查未完成的 Agent 確認狀態失敗:', e);
    }
}

let lastAutoLabelStatus = "idle";
async function checkAutoLabelStatus() {
    try {
        const res = await fetch('/api/ai/auto_label_status');
        const data = await res.json();
        
        if (data.status === "finished" && lastAutoLabelStatus === "running") {
            document.getElementById('auto-label-panel').style.display = 'none';
            showAlert(`✅ 自動標記完成！已成功為「${data.dataset}」標記了 ${data.labeled_count} 張圖片。`, "success");
            
            const labelingIframe = document.getElementById('iframe-labeling');
            if (labelingIframe && labelingIframe.contentWindow) {
                if (activeDataset && activeDataset.name === data.dataset) {
                    labelingIframe.contentWindow.postMessage({ type: 'DATASET_SELECTED', dataset: activeDataset }, '*');
                }
            }
        } else if (data.status === "error" && lastAutoLabelStatus === "running") {
            document.getElementById('auto-label-panel').style.display = 'none';
            showAlert(`❌ 自動標記失敗: ${data.error}`, "error");
        } else if (data.status === "running") {
            // 更新懸浮面板進度
            const panel = document.getElementById('auto-label-panel');
            panel.style.display = 'block';
            
            const processed = data.processed || 0;
            const total = data.total || 0;
            const labeled = data.labeled_count || 0;
            const percent = total > 0 ? Math.round((processed / total) * 100) : 0;
            
            document.getElementById('auto-label-bar').style.width = percent + '%';
            document.getElementById('auto-label-percent').innerText = percent + '%';
            document.getElementById('auto-label-stats').innerText = `掃描進度: ${processed} / ${total}`;
            document.getElementById('auto-label-count').innerText = `已標記: ${labeled} 張`;
        }
        
        lastAutoLabelStatus = data.status;
    } catch (e) {
        console.error("Auto label status poll error:", e);
    }
}

function switchPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById('page-' + pageId);
    if (pageEl) {
        pageEl.classList.add('active');
        setTimeout(() => { pageEl.style.opacity = '1'; }, 10);
    }
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('onclick').includes(pageId)) item.classList.add('active');
    });
    if (pageId === 'train') loadTrainDatasets();
    
    // 記憶當前頁面
    localStorage.setItem('ai_detector_active_page', pageId);
}

function showAlert(message, type = 'info') {
    const alertBox = document.getElementById('alert-box');
    alertBox.innerText = message;
    alertBox.style.display = 'block';
    alertBox.style.background = (type === 'error') ? '#ff4757' : '#111';
    setTimeout(() => { alertBox.style.display = 'none'; }, 5000);
}

function openLightbox(src) {
    if (!src) return;
    const overlay = document.getElementById('lightbox-overlay');
    document.getElementById('lightbox-img').src = src;
    overlay.style.display = 'flex';
}
function closeLightbox() {
    document.getElementById('lightbox-overlay').style.display = 'none';
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

// ------------------------------
// 其他功能保持簡潔
// ------------------------------
async function updateSystemMonitor() {
    try {
        const res = await fetch('/api/system/info');
        const data = await res.json();
        
        // 更新 CPU & RAM
        const cpuEl = document.getElementById('sys-cpu');
        cpuEl.innerText = data.cpu || "-";
        cpuEl.title = data.cpu || "";
        document.getElementById('sys-ram').innerText = data.ram || "-";
        
        // 更新模式與 GPU 資訊
        const modeEl = document.getElementById('sys-mode');
        const gpuEl = document.getElementById('sys-gpu');
        
        if (data.cuda) {
            modeEl.innerText = "GPU ACCEL";
            modeEl.style.background = "#10b981"; // 綠色代表效能
            gpuEl.innerText = data.gpu || "Generic GPU";
            gpuEl.style.color = "#34d399";
        } else {
            modeEl.innerText = "CPU ONLY";
            modeEl.style.background = "#475569"; // 灰色代表基礎模式
            gpuEl.innerText = "No NVIDIA GPU detected";
            gpuEl.style.color = "#64748b";
        }
    } catch (e) {
        console.error("Monitor error:", e);
    }
}

// ------------------------------
// 訓練邏輯
// ------------------------------

function initChart() {
    const ctx = document.getElementById('trainChart').getContext('2d');
    if (trainChart) trainChart.destroy();

    trainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // 輪次標籤
            datasets: [
                {
                    label: 'mAP50 (%)',
                    data: [],
                    borderColor: '#3b82f6',
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Loss',
                    data: [],
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    type: 'linear', 
                    position: 'left', 
                    min: 0, 
                    max: 100, 
                    title: { display: true, text: 'mAP (%)' } 
                },
                y1: { 
                    type: 'linear', 
                    position: 'right', 
                    min: 0, 
                    grid: { drawOnChartArea: false }, 
                    title: { display: true, text: 'Loss' } 
                }
            },
            plugins: {
                legend: { display: true, position: 'top' }
            }
        }
    });
}

async function loadTrainDatasets() {
    try {
        const res = await fetch('/api/datasets/list');
        const data = await res.json();
        const select = document.getElementById('train-dataset-select');
        const current = select.value;
        select.innerHTML = '<option value="">-- 請選擇已匯入的資料集 --</option>';
        if (data.status === 'success') {
            data.datasets.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.name;
                opt.textContent = `${ds.name} (${ds.imgCount || 0} 張圖片)`;
                select.appendChild(opt);
            });
            if (current) select.value = current;
        }
    } catch (e) { console.error('Load datasets error:', e); }
}

async function startTraining() {
    if (trainInterval) return;

    const startBtn = document.getElementById('start-train-btn');
    const datasetName = document.getElementById('train-dataset-select').value;

    if (!datasetName) {
        showAlert("請先選擇要訓練的資料集", "error");
        return;
    }

    startBtn.disabled = true;
    startBtn.innerText = "啟動中...";

    const formData = new FormData();
    formData.append('dataset_name', datasetName);
    formData.append('model_name', document.getElementById('train-model-name').value);
    formData.append('variant', document.getElementById('train-model-variant').value);
    formData.append('epochs', document.getElementById('train-epochs').value);
    formData.append('patience', document.getElementById('train-patience').value);
    formData.append('batch', document.getElementById('train-batch').value);
    formData.append('imgsz', document.getElementById('train-imgsz').value);

    try {
        const res = await fetch('/api/train/start', { method: 'POST', body: formData });
        const result = await res.json();

        if (result.status === "error") {
            showAlert(result.message, "error");
            startBtn.disabled = false;
            startBtn.innerText = "啟動訓練";
            return;
        }

        document.getElementById('training-progress-card').style.display = 'block';
        initChart();

        trainInterval = setInterval(async () => {
            if (isPaused) return;
            try {
                const r = await fetch('/api/train/status');
                const d = await r.json();
                
                const progressBar = document.getElementById('train-progress-bar');
                const statusText = document.getElementById('train-status-text');
                
                if (d.status === "training") {
                    const p = Math.round((d.current_epoch / d.total_epochs) * 100);
                    progressBar.style.width = `${p}%`;
                    statusText.innerText = `訓練進行中: 第 ${d.current_epoch} / ${d.total_epochs} 輪 (${p}%)`;
                    
                    if (d.metrics && d.metrics.length > 0) {
                        trainChart.data.labels = d.metrics.map(m => m.epoch);
                        trainChart.data.datasets[0].data = d.metrics.map(m => m.map);
                        trainChart.data.datasets[1].data = d.metrics.map(m => m.loss);
                        trainChart.update();
                    }
                } else if (d.status === "completed" || d.status === "error") {
                    clearInterval(trainInterval);
                    trainInterval = null;
                    startBtn.disabled = false;
                    startBtn.innerText = "啟動訓練";
                    statusText.innerText = d.status === "completed" ? "訓練已完成" : "訓練發生錯誤";
                    if (d.status === "completed") finishTraining("訓練成功完成", "success");
                }
            } catch (e) { console.error(e); }
        }, 2000);
    } catch (e) {
        showAlert("啟動訓練失敗: " + e.message, "error");
        startBtn.disabled = false;
        startBtn.innerText = "啟動訓練";
    }
}

function finishTraining(msg, type) {
    showAlert(msg, type);
    // 顯示評價
    document.getElementById('train-evaluation').style.display = 'block';
}

function terminateTraining() {
    if (confirm("確定終止？")) {
        fetch('/api/train/stop', { method: 'POST' });
        clearInterval(trainInterval);
        trainInterval = null;
        document.getElementById('start-train-btn').disabled = false;
        document.getElementById('train-status-text').innerText = "❌ 訓練已終止";
    }
}

function togglePause() {
    isPaused = !isPaused;
    document.getElementById('pause-train-btn').innerText = isPaused ? "恢復" : "暫停";
}

// ------------------------------
// 驗證與同步邏輯 [RESTORED]
// ------------------------------
function syncParams(el, selector) {
    const val = (el.type === 'checkbox') ? el.checked : el.value;
    document.querySelectorAll(selector).forEach(target => {
        if (target !== el) {
            if (target.type === 'checkbox') target.checked = val;
            else target.value = val;
            
            // 更新數值顯示
            const display = target.parentElement.querySelector('span[class$="-val"]');
            if (display) display.innerText = val;
        }
    });
    
    // 如果是當前操作的元素，也要更新它的顯示
    const currentDisplay = el.parentElement.querySelector('span[class$="-val"]');
    if (currentDisplay) currentDisplay.innerText = val;
}

function handleFilesUpload(input, type) {
    const files = Array.from(input.files);
    if (type === 'single') {
        singleFiles = files;
        if (files.length > 0) {
            const reader = new FileReader();
            reader.onload = e => document.getElementById('prediction-img-single').src = e.target.result;
            reader.readAsDataURL(files[0]);
        }
    } else {
        batchFiles = files.filter(f => f.name.match(/\.(jpg|jpeg|png)$/i));
        showAlert(`已選擇 ${batchFiles.length} 張影像`, "success");
    }
}

async function startVerification(type) {
    const model = document.querySelector('.model-selector').value;
    const imgsz = document.querySelector('.imgsz-selector').value;
    const conf = document.querySelector('.conf-range').value;
    const iou = document.querySelector('.iou-range').value;
    const augment = document.querySelector('.augment-toggle').checked;

    if (type === 'single' && singleFiles.length === 0) return showAlert("請先選擇影像", "error");

    const formData = new FormData();
    formData.append('model_name', model);
    formData.append('imgsz', imgsz);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('augment', augment);
    
    if (type === 'single') formData.append('file', singleFiles[0]);

    showAlert("啟動檢測中...", "info");
    try {
        const res = await fetch('/api/verify/single', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('prediction-img-single').src = data.image_url + '?t=' + Date.now();
            showAlert("檢測完成", "success");
        } else {
            showAlert(data.message, "error");
        }
    } catch (e) { showAlert("連線失敗", "error"); }
}

async function startBatchVerification() {
    if (batchFiles.length === 0) return showAlert("請先選擇資料夾", "error");
    // ... 批量邏輯實作 ...
    showAlert("批量檢測功能執行中...", "info");
}

function toggleTooltip(id) {
    const box = document.getElementById('tooltip-' + id);
    box.style.display = (box.style.display === 'block') ? 'none' : 'block';
}

function clearFiles() {
    if (confirm("確定清空伺服器快取？")) {
        fetch('/api/clear', { method: 'POST' }).then(() => showAlert("快取已清除", "success"));
    }
}

// ------------------------------
// 初始化
// ------------------------------
window.onload = () => {
    switchPage('datasets');
    setInterval(updateSystemMonitor, 30000);
};

// ------------------------------
// AI Chat Assistant Logic
// ------------------------------
let chatMessages = [];

function onAiModelSelectChange() {
    // Manager Agent 底層要用哪個 LLM，用第二個下拉選單另外選——切到別的選項時就隱藏，
    // 避免使用者以為一般聊天視窗也能選 Manager Agent 的引擎
    const modelType = document.getElementById('ai-model-select').value;
    const engineBar = document.getElementById('manager-agent-engine-bar');
    engineBar.style.display = modelType === 'manager-agent' ? 'flex' : 'none';
}

function toggleChat() {
    const chatWindow = document.getElementById('ai-chat-window');
    const btn = document.getElementById('ai-chat-btn');
    if (chatWindow.style.display === 'none') {
        chatWindow.style.display = 'flex';
        btn.style.transform = 'scale(0.9)';
    } else {
        chatWindow.style.display = 'none';
        btn.style.transform = 'scale(1)';
    }
}

async function sendChatMessage() {
    const input = document.getElementById('ai-chat-input');
    const text = input.value.trim();
    if (!text) return;
    
    const modelType = document.getElementById('ai-model-select').value;
    console.log(">>> [Browser] Sending chat request to platform:", modelType);
    appendMessage(text, 'user');
    input.value = '';
    
    // Add context if needed (e.g., current dataset info)
    const contextData = {
        activeDataset: activeDataset ? activeDataset.name : "None",
        currentTrainEpoch: trainChart && trainChart.data.labels.length > 0 ? trainChart.data.labels[trainChart.data.labels.length-1] : "Not training"
    };

    chatMessages.push({ role: 'user', content: text });

    const loadingId = appendMessage(`[${modelType.toUpperCase()}] 思考中...`, 'ai', true);

    // Manager Agent 走獨立的 /api/agent/chat（LangGraph + Tool，SSE 串流），跟原本的
    // /api/ai/chat（Gemini/Groq/Local 的一般對話，整包 JSON）分開，互不影響。Manager
    // Agent 底層要用哪個 LLM 是另一個下拉選單決定的（manager-agent-engine-select），
    // 跟一般聊天視窗的 modelType 是兩件事，不要混在一起。
    if (modelType === 'manager-agent') {
        await sendManagerAgentMessage(text, loadingId);
        return;
    }

    const requestBody = { messages: chatMessages, context_data: contextData, model_type: modelType };

    try {
        const res = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        const data = await res.json();

        removeMessage(loadingId);

        if (data.status === 'success') {
            let reply = data.reply;

            // 檢查是否有健康度數據標籤 [HEALTH_DATA: {json}]
            const healthMatch = reply.match(/\[HEALTH_DATA: ({.*?})\]/);
            if (healthMatch) {
                const healthJson = JSON.parse(healthMatch[1]);
                reply = reply.replace(healthMatch[0], ""); // 移除標籤，保留文字
                appendMessage(reply, 'ai');
                appendHealthCard(healthJson);
            } else {
                appendMessage(reply, 'ai');
            }

            chatMessages.push({ role: 'assistant', content: reply });
        } else {
            appendMessage("發生錯誤: " + data.message, 'error');
        }
    } catch (e) {
        removeMessage(loadingId);
        appendMessage("連線失敗", 'error');
    }
}

// Manager Agent 的對話記憶（LangGraph checkpointer）用 thread_id 對應到同一份
// state——存在 sessionStorage 而不是 localStorage，分頁關閉就清除，符合「這次
// 用著有記憶、下次進站重置」；重新整理頁面不會清（瀏覽器對 sessionStorage 的
// 行為本來就是這樣），要真的開新對話請按「開新對話」按鈕。
const MANAGER_AGENT_THREAD_KEY = 'manager_agent_thread_id';

function getOrCreateThreadId() {
    let threadId = sessionStorage.getItem(MANAGER_AGENT_THREAD_KEY);
    if (!threadId) {
        threadId = crypto.randomUUID();
        sessionStorage.setItem(MANAGER_AGENT_THREAD_KEY, threadId);
    }
    return threadId;
}

function resetManagerAgentThread() {
    sessionStorage.setItem(MANAGER_AGENT_THREAD_KEY, crypto.randomUUID());
    chatMessages.length = 0;
    document.getElementById('ai-chat-history').innerHTML = '';
}

// /api/agent/chat、/api/agent/resume 回應都是 SSE 串流（text/event-stream），逐個
// trace step 邊跑邊塞進決策過程卡片（createTraceCard/appendTraceStep），不用等整個
// Plan→Act→Observe→Decide 迴圈跑完才看到東西；中途失敗也能看到已經跑出來的步驟跟
// 錯誤原因。兩個端點的 SSE frame 格式一樣，解析邏輯抽成這個共用函式，resume 時
// 沿用同一張決策過程卡片（傳入既有的 traceRefs）繼續往下疊，不會另開一張新卡片。
async function consumeAgentStream(res, traceRefs) {
    let finalReply = null;
    let sawError = false;
    let interruptPayload = null;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const rawEvent = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            if (!rawEvent.startsWith('data: ')) continue;

            const payload = JSON.parse(rawEvent.slice(6));
            if (payload.type === 'step') {
                if (!traceRefs) traceRefs = createTraceCard();
                appendTraceStep(traceRefs, payload);
            } else if (payload.type === 'final') {
                finalReply = payload.reply;
                if (traceRefs) finishTraceCard(traceRefs, payload.usage, payload.llm_calls);
            } else if (payload.type === 'error') {
                sawError = true;
                if (!traceRefs) traceRefs = createTraceCard();
                appendTraceError(traceRefs, payload.message);
            } else if (payload.type === 'interrupt') {
                interruptPayload = payload.payload;
            }
        }
    }

    return { traceRefs, finalReply, sawError, interruptPayload };
}

// 卡在 interrupt()（目前只有訓練前確認會觸發）時，用專案裡既有的瀏覽器原生
// confirm() 慣例（跟「終止訓練」「清空快取」同一套模式，不另外做卡片式按鈕 UI）
// 問使用者要不要繼續，答案立刻送去 /api/agent/resume 接著跑；用迴圈是為了以後
// 如果同一輪內有不只一個 interrupt 點，也能一樣接住，不用另外改這段邏輯。
async function finishOrResumeAgentTurn(result, threadId, modelType) {
    let { traceRefs, finalReply, sawError, interruptPayload } = result;

    while (interruptPayload) {
        const approved = confirm(interruptPayload.reason || "確認要繼續嗎？");
        const res = await fetch('/api/agent/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: threadId, model_type: modelType, approved })
        });
        const next = await consumeAgentStream(res, traceRefs);
        traceRefs = next.traceRefs;
        finalReply = next.finalReply;
        sawError = next.sawError;
        interruptPayload = next.interruptPayload;
    }

    if (finalReply !== null) {
        let reply = finalReply;
        const healthMatch = reply.match(/\[HEALTH_DATA: ({.*?})\]/);
        if (healthMatch) {
            const healthJson = JSON.parse(healthMatch[1]);
            reply = reply.replace(healthMatch[0], "");
            appendMessage(reply, 'ai');
            appendHealthCard(healthJson);
        } else {
            appendMessage(reply, 'ai');
        }
        chatMessages.push({ role: 'assistant', content: reply });
    } else if (sawError) {
        appendMessage("執行中發生錯誤，詳情請展開上方決策過程", 'error');
    }
}

async function sendManagerAgentMessage(text, loadingId) {
    const modelType = document.getElementById('manager-agent-engine-select').value;
    const threadId = getOrCreateThreadId();
    const requestBody = { message: text, model_type: modelType, thread_id: threadId };

    try {
        const res = await fetch('/api/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        removeMessage(loadingId);

        const result = await consumeAgentStream(res, null);
        await finishOrResumeAgentTurn(result, threadId, modelType);
    } catch (e) {
        removeMessage(loadingId);
        appendMessage("連線失敗", 'error');
    }
}

function appendHealthCard(data) {
    const container = document.getElementById('ai-chat-history');
    const card = document.createElement('div');
    card.style.cssText = "align-self: flex-start; background: #0f172a; border: 1px solid #3b82f6; padding: 16px; border-radius: 12px; width: 80%; margin-top: 10px;";
    
    const chartId = 'chat-chart-' + Date.now();
    card.innerHTML = `
        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px; display:flex; align-items:center; gap:5px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            AI 診斷報告: ${data.coverage}% 標記率
        </div>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="width: 100px; height: 100px;">
                <canvas id="${chartId}"></canvas>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 1.2rem; font-weight: 800; color: #3b82f6;">Score: ${data.score}</div>
                <div style="font-size: 0.7rem; color: #94a3b8;">平衡度: ${data.balance}%</div>
                <div style="font-size: 0.7rem; color: #94a3b8;">總數: ${data.total_images} 張</div>
            </div>
        </div>
    `;
    
    container.appendChild(card);
    container.scrollTop = container.scrollHeight;
    
    // 延遲一點點渲染圖表，確保 DOM 已經掛載
    setTimeout(() => {
        const ctx = document.getElementById(chartId).getContext('2d');
        const labels = Object.values(data.class_distribution).map(v => v.name);
        const counts = Object.values(data.class_distribution).map(v => v.count);
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: labels.map((_, i) => colors[i % colors.length]),
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
            }
        });
    }, 100);
}

const MANAGER_AGENT_NODE_LABELS = {
    planner: '🗺️ Planner（規劃）',
    manager: '🧭 Manager（決策）',
    observer: '🔍 Observer（觀測）',
    data_agent: '📁 Data Agent',
    model_agent: '🤖 Model Agent',
    evaluation_agent: '📊 Evaluation Agent'
};

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value === null || value === undefined ? '' : String(value);
    return div.innerHTML;
}

// 把 /api/agent/chat 串流回來的單個 trace step（Planner 的計畫、Manager 的
// 路由決策、Observer 的判讀、子 Agent 呼叫的工具）轉成一段 HTML，給
// appendTraceStep 逐步塞進卡片——不然使用者只看得到最終一句回覆，看不出
// 三個 Agent 實際怎麼協作。
function renderTraceStepHtml(step, index) {
    const label = MANAGER_AGENT_NODE_LABELS[step.node] || step.node;
    let inner = `<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.15);">`;
    inner += `<div style="color:#e2e8f0; font-weight:600;">#${index} ${escapeHtml(label)}</div>`;

    if (step.decision) {
        inner += `<div>→ 交給：<b>${escapeHtml(step.decision)}</b></div>`;
    }
    if (step.plan && step.plan.goal) {
        inner += `<div>🎯 目標：${escapeHtml(step.plan.goal)}</div>`;
        if (step.plan.steps && step.plan.steps.length) {
            inner += '<ul style="margin:4px 0 0 16px; padding:0;">' +
                step.plan.steps.map(s => `<li>[${escapeHtml(s.status)}] ${escapeHtml(s.description)}（${escapeHtml(s.agent)}）</li>`).join('') +
                '</ul>';
        }
    }
    if (step.tool_calls && step.tool_calls.length) {
        inner += step.tool_calls.map(tc => {
            if (tc.tool) {
                return `<div>🔧 呼叫工具：<b>${escapeHtml(tc.tool)}</b>（${escapeHtml(JSON.stringify(tc.args))}）</div>`;
            }
            return `<div>📥 工具回傳（${escapeHtml(tc.tool_result)}）：${escapeHtml(typeof tc.output === 'string' ? tc.output : JSON.stringify(tc.output))}</div>`;
        }).join('');
    }
    if (step.report) {
        inner += `<div>📝 回報：status=<b>${escapeHtml(step.report.status)}</b>` +
            (step.report.reason_code ? ` reason_code=${escapeHtml(step.report.reason_code)}` : '') +
            `<br>理由：${escapeHtml(step.report.reason)}` +
            (step.report.suggestion ? `<br>建議：${escapeHtml(step.report.suggestion)}` : '') +
            `</div>`;
    }
    if (step.evaluation) {
        inner += `<div>🔍 判讀：verdict=<b>${escapeHtml(step.evaluation.verdict)}</b>` +
            `，有進展=${step.evaluation.progress_since_last_round ? '是' : '否'}` +
            (step.evaluation.contradiction ? '，⚠️ 偵測到矛盾' : '') +
            `</div>`;
    }
    if (step.text) {
        inner += `<div>💬 ${escapeHtml(step.text)}</div>`;
    }
    if (step.usage) {
        inner += `<div style="color:#64748b;">🔢 tokens：input ${step.usage.input_tokens} / output ${step.usage.output_tokens}</div>`;
    }
    inner += `</div>`;
    return inner;
}

// 建立空的決策過程卡片，跑到哪一步就用 appendTraceStep 補進去，不用等全部
// 跑完才一次性渲染——預設展開（open），因為需求就是「跑到哪就看到哪」。
function createTraceCard() {
    const container = document.getElementById('ai-chat-history');

    const wrapper = document.createElement('details');
    wrapper.open = true;
    wrapper.style.cssText = "align-self: flex-start; width: 92%; margin-top: 6px; background: #0f172a; border: 1px solid rgba(59,130,246,0.3); border-radius: 10px; padding: 8px 12px; font-size: 0.72rem; color: #94a3b8; overflow-wrap: anywhere; word-break: break-word;";

    const summary = document.createElement('summary');
    summary.style.cssText = "cursor: pointer; color: #60a5fa; font-weight: 600; outline: none;";
    summary.textContent = '🔬 決策過程（執行中…）';
    wrapper.appendChild(summary);

    const stepsContainer = document.createElement('div');
    wrapper.appendChild(stepsContainer);

    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;

    return { wrapper, summary, stepsContainer, stepCount: 0 };
}

function appendTraceStep(refs, step) {
    refs.stepCount += 1;
    const stepDiv = document.createElement('div');
    stepDiv.innerHTML = renderTraceStepHtml(step, refs.stepCount);
    refs.stepsContainer.appendChild(stepDiv);
    refs.summary.textContent = `🔬 決策過程（共 ${refs.stepCount} 步，點開查看）`;

    const container = document.getElementById('ai-chat-history');
    container.scrollTop = container.scrollHeight;
}

function appendTraceError(refs, message) {
    const errDiv = document.createElement('div');
    errDiv.style.cssText = "margin-top:8px; padding-top:8px; border-top:1px solid rgba(239,68,68,0.3); color:#f87171;";
    errDiv.textContent = `⚠️ 執行中發生錯誤：${message}`;
    refs.stepsContainer.appendChild(errDiv);
    refs.summary.textContent = `🔬 決策過程（共 ${refs.stepCount} 步，中途失敗）`;
}

// 跑完後在卡片底部補一行總計，讓使用者不用自己把每一步的 token 加起來。
// usage/llmCalls 可能是 undefined（模型沒填 usage_metadata，例如本機 Ollama）。
function finishTraceCard(refs, usage, llmCalls) {
    if (!usage || !usage.total_tokens) return;
    const summaryDiv = document.createElement('div');
    summaryDiv.style.cssText = "margin-top:8px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.15); color:#64748b;";
    summaryDiv.textContent = `📊 共 ${llmCalls || 0} 次 LLM 呼叫，${usage.total_tokens} tokens（input ${usage.input_tokens} / output ${usage.output_tokens}）`;
    refs.stepsContainer.appendChild(summaryDiv);
}

function appendMessage(text, sender, isLoading = false) {
    const container = document.getElementById('ai-chat-history');
    const msgDiv = document.createElement('div');
    const id = 'msg-' + Date.now();
    msgDiv.id = id;
    
    msgDiv.style.padding = '10px 14px';
    msgDiv.style.borderRadius = '12px';
    msgDiv.style.maxWidth = '85%';
    msgDiv.style.lineHeight = '1.4';
    
    if (sender === 'user') {
        msgDiv.style.alignSelf = 'flex-end';
        msgDiv.style.background = '#3b82f6';
        msgDiv.style.color = 'white';
    } else if (sender === 'error') {
        msgDiv.style.alignSelf = 'flex-start';
        msgDiv.style.background = 'rgba(239, 68, 68, 0.2)';
        msgDiv.style.border = '1px solid rgba(239, 68, 68, 0.5)';
        msgDiv.style.color = '#fca5a5';
    } else {
        msgDiv.style.alignSelf = 'flex-start';
        msgDiv.style.background = 'rgba(59, 130, 246, 0.1)';
        msgDiv.style.border = '1px solid rgba(59, 130, 246, 0.2)';
        msgDiv.style.color = '#e2e8f0';
        if (isLoading) msgDiv.style.opacity = '0.5';
    }
    
    msgDiv.innerText = text;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function quickPrompt(text) {
    document.getElementById('ai-chat-input').value = text;
    sendChatMessage();
}

// --- 整合驗證中心邏輯 ---
let unifiedFiles = [];

function handleVerifyUpload(input, type) {
    const files = Array.from(input.files);
    const imageExtensions = /\.(jpg|jpeg|png|bmp|webp)$/i;
    unifiedFiles = files.filter(f => f.name.match(imageExtensions));
    
    // 清理另一個輸入框的選擇
    if (type === 'folder') {
        document.getElementById('file-input-verify-files').value = '';
    } else {
        document.getElementById('file-input-verify-folder').value = '';
    }
    
    const statusText = document.getElementById('verify-status-text');
    if (unifiedFiles.length > 0) {
        if (type === 'folder') {
            statusText.innerText = `📁 已選資料夾，含 ${unifiedFiles.length} 張圖片`;
        } else {
            statusText.innerText = `📸 已選 ${unifiedFiles.length} 張圖片`;
        }
        statusText.style.color = '#3b82f6';
    } else {
        statusText.innerText = '未選擇任何有效影像';
        statusText.style.color = '#ef4444';
    }
}

async function startUnifiedVerification() {
    const model = document.getElementById('verify-model-select').value;
    const imgsz = document.getElementById('verify-imgsz-select').value;
    const conf = document.getElementById('verify-conf-range').value;
    const iou = document.getElementById('verify-iou-range').value;
    const tta = document.getElementById('verify-tta-toggle').checked;
    const container = document.getElementById('verify-results-container');

    if (unifiedFiles.length === 0) {
        showAlert("請先選擇圖片或資料夾", "error");
        return;
    }

    container.innerHTML = '<div style="color:#3b82f6; grid-column: 1/-1; text-align:center; padding:20px;">AI 正在分析中...</div>';

    try {
        const formData = new FormData();
        unifiedFiles.forEach(f => formData.append('files', f));
        formData.append('model', model);
        formData.append('imgsz', imgsz);
        formData.append('conf', conf);
        formData.append('iou', iou);
        formData.append('augment', tta);

        const res = await fetch('/api/verify/files', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'error') throw new Error(data.message);
        
        renderVerifyResults(data.results);
    } catch (err) {
        showAlert("檢測失敗: " + err.message, "error");
        container.innerHTML = '';
    }
}

function renderVerifyResults(results) {
    const container = document.getElementById('verify-results-container');
    if (results.length === 0) {
        container.innerHTML = '<div style="color:#94a3b8; grid-column: 1/-1; text-align:center; padding:20px;">未偵測到任何結果</div>';
        return;
    }

    container.innerHTML = results.map(r => `
        <div class="card" style="padding:0; overflow:hidden; background: rgba(30, 41, 59, 0.5);">
            <img src="${r.url}?t=${Date.now()}" style="width:100%; height:150px; object-fit:cover; cursor:pointer;" onclick="window.open(this.src)">
            <div style="padding:10px; font-size:0.75rem;">
                <div style="color:#fff; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${r.name}</div>
                <div style="color:#64748b; margin-top:4px;">目標數: ${r.objects ? r.objects.length : 0}</div>
                ${r.objects && r.objects.length > 0 ? `
                    <div style="margin-top:8px; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px; max-height:100px; overflow-y:auto; display:flex; flex-direction:column; gap:4px;">
                        ${r.objects.map(obj => {
                            const [x1, y1, x2, y2] = obj.bbox.map(coord => Math.round(coord));
                            return `
                                <div style="display:flex; justify-content:space-between; color:#cbd5e1; font-size:0.65rem;">
                                    <span style="color:#60a5fa; font-weight:600;">${obj.name}</span>
                                    <span>${(obj.confidence * 100).toFixed(0)}% [${x1},${y1},${x2},${y2}]</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                ` : ''}
            </div>
        </div>
    `).join('');
}
async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const select = document.getElementById('verify-model-select');
        const diagSelect = document.getElementById('diagnosis-model-select');
        
        const current = select.value;
        const currentDiag = diagSelect ? diagSelect.value : '';
        
        select.innerHTML = '';
        if (diagSelect) diagSelect.innerHTML = '<option value="">-- 請選擇要診斷的模型 --</option>';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                select.appendChild(opt);
                
                if (diagSelect) {
                    const optDiag = document.createElement('option');
                    optDiag.value = m;
                    optDiag.textContent = m;
                    diagSelect.appendChild(optDiag);
                }
            });
            if (current && data.models.includes(current)) {
                select.value = current;
            } else if (data.models.includes('yolov9m.pt')) {
                select.value = 'yolov9m.pt';
            }
            
            if (diagSelect && currentDiag && data.models.includes(currentDiag)) {
                diagSelect.value = currentDiag;
            }
        } else {
            select.innerHTML = '<option value="">無可用模型</option>';
        }
    } catch (e) {
        console.error('Load models error:', e);
    }
}

// 頁面載入時初始化模型選單
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
});

async function loadDiagnosisCharts() {
    // 從診斷頁面獲取當前選用的模型，並推導出訓練資料夾名稱
    const selectedModel = document.getElementById('diagnosis-model-select').value;
    if (!selectedModel) {
        showAlert("請先選擇要診斷的模型", "error");
        return;
    }
    
    // 假設檔名格式為: my_model_best.pt，則資料夾名稱為 my_model
    let folderName = selectedModel.replace('_best.pt', '').replace('.pt', '');

    const charts = [
        { id: 'chart-confusion', name: 'confusion_matrix.png' },
        { id: 'chart-f1', name: 'BoxF1_curve.png' },
        { id: 'chart-pr', name: 'BoxPR_curve.png' }
    ];

    for (const chart of charts) {
        const el = document.getElementById(chart.id);
        if (el) {
            // 帶上 model 參數
            el.src = `/api/model/evaluation/charts/${chart.name}?model=${encodeURIComponent(folderName)}&t=${Date.now()}`;
            el.onerror = () => {
                console.warn(`Chart ${chart.name} not found.`);
                el.alt = "尚未生成此報表 (請先完成模型訓練)";
            };
        }
    }
    showAlert(`正在載入 ${folderName} 的診斷報告...`, "success");
}
