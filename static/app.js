// Custom JavaScript for AI Photo Site

// ============================================================================
// Wizard Steps Data (Before you start)
// ============================================================================
const wizardSteps = [
    {
        title: "İyi fotoğraf böyle olur",
        badge: "Kontrol Listesi",
        bullets: [
            "Gözler açık ve kameraya bakıyor",
            "Gözlük veya şapka yok (yansıma yok)",
            "Fotoğraf net ve düz",
            "Yüz tam görünüyor, çok yakın/uzak değil"
        ],
        type: "checklist"
    },
    {
        title: "Gözler açık, kameraya bak",
        badge: "Önemli",
        bullets: [
            "Gözleriniz tamamen açık olmalı ve doğrudan kameraya bakmalısınız",
            "Kapalı gözler veya başka yöne bakmak AI tarafından düzeltilemez",
            "Gözler kapalıysa veya başka yöne bakıyorsa, fotoğrafı yeniden çekmeniz gerekecektir"
        ],
        type: "warning"
    },
    {
        title: "Gözlük/şapka yok",
        badge: "Dikkat",
        bullets: [
            "Fotoğrafta gözlük veya şapka olmamalıdır",
            "Gözlük camında yansıma varsa bu da kabul edilmez",
            "Güneş gözlüğü kesinlikle kullanılmamalı",
            "Şapka, bere veya başörtüsü (yüzü kapatmayan hariç) olmamalı"
        ],
        type: "info"
    },
    {
        title: "Net ve düz dur",
        badge: "Kurallar",
        bullets: [
            "Fotoğraf net olmalı ve kafa düz durmalıdır",
            "Bulanıklık veya odak hatası kabul edilmez",
            "Kafa çok sağa/sola dönük olmamalı",
            "Çok yakın çekilmiş (sadece yüz görünüyor) veya çok uzak çekilmiş (yüz çok küçük) olmamalı"
        ],
        type: "info"
    },
    {
        title: "Biz neyi otomatik düzeltiyoruz",
        badge: "Otomatik Düzeltme",
        bullets: [
            "Arka plan: Beyaz arka plana çevrilir",
            "Işık dengesi: Fotoğrafın aydınlatması optimize edilir",
            "Küçük eğimler: Hafif açısal hatalar düzeltilir"
        ],
        type: "success"
    }
];

// ============================================================================
// Processing Steps & Timing
// ============================================================================
const MIN_PROCESSING_MS = 5000; // Minimum processing süresi (5 saniye)
const STEP_POINTS = [
    { key: "crop", t: 1200 },
    { key: "bg_remove", t: 2400 },
    { key: "resize", t: 3600 },
    { key: "analyze", t: 4800 }
];

const processingSteps = [
    { name: "Kırpma", key: "crop" },
    { name: "Arka plan kaldırma", key: "bg_remove" },
    { name: "Yeniden boyutlandırma", key: "resize" },
    { name: "Analiz", key: "analyze" }
];

// ============================================================================
// Checklist Items
// ============================================================================
const checklistItems = [
    { key: "face_detected", label: "Yüz tanındı" },
    { key: "single_face", label: "Fotoğrafta yalnızca bir yüz olmalı" },
    { key: "min_size", label: "Minimum boyut" },
    { key: "aspect_ratio_ok", label: "Fotoğraf oranları doğru" }
];

// ============================================================================
// State
// ============================================================================
let currentWizardStep = 0;
let currentJobId = null;
let currentImageUrl = null;
let currentPreviewUrl = null;
let processingStepIndex = 0;
let processingInterval = null;
let pollingInterval = null;
let processingStart = 0;
let backendDonePayload = null;
let stepStates = {}; // { crop: "pending|processing|done", ... }
let scanRunning = false;
let scanRaf = null;
let uiTimer = null;

// ============================================================================
// DOM Elements
// ============================================================================
const wizardModal = document.getElementById('wizardModal');
const processingModal = document.getElementById('processingModal');
const resultModal = document.getElementById('resultModal');
const uploadForm = document.getElementById('uploadForm');
const photoInput = document.getElementById('photoInput');

// ============================================================================
// Wizard Modal Functions
// ============================================================================
function openWizardModal() {
    if (!wizardModal) return;
    currentWizardStep = 0;
    renderWizardStep();
    renderWizardStepIndicator();
    wizardModal.classList.remove('hidden');
}

function closeWizardModal() {
    if (!wizardModal) return;
    wizardModal.classList.add('hidden');
}

function renderWizardStep() {
    const wizardStepContent = document.getElementById('wizardStepContent');
    if (!wizardStepContent) return;
    const step = wizardSteps[currentWizardStep];
    
    let html = `
        <div class="space-y-4">
            <div class="flex items-center gap-2 mb-4">
                <h3 class="text-lg font-semibold text-gray-800">${step.title}</h3>
                ${step.badge ? `<span class="px-2 py-1 text-xs rounded bg-blue-100 text-blue-800">${step.badge}</span>` : ''}
            </div>
            <ul class="space-y-2">
    `;
    
    step.bullets.forEach(bullet => {
        html += `
            <li class="flex items-start space-x-2">
                <span class="text-blue-500 mt-1">•</span>
                <p class="text-gray-700">${bullet}</p>
            </li>
        `;
    });
    
    html += `
            </ul>
        </div>
    `;
    
    wizardStepContent.innerHTML = html;
    updateWizardButtons();
}

function renderWizardStepIndicator() {
    const indicator = document.getElementById('stepIndicator');
    if (!indicator) return;
    
    indicator.innerHTML = '';
    wizardSteps.forEach((_, index) => {
        const dot = document.createElement('div');
        dot.className = `w-2 h-2 rounded-full ${index === currentWizardStep ? 'bg-blue-600' : 'bg-gray-300'}`;
        indicator.appendChild(dot);
    });
}

function updateWizardButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const nextButtonContainer = document.getElementById('nextButtonContainer');
    
    if (!prevBtn || !nextBtn || !nextButtonContainer) return;
    
    prevBtn.classList.toggle('hidden', currentWizardStep === 0);
    
    if (currentWizardStep === wizardSteps.length - 1) {
        nextBtn.classList.add('hidden');
        nextButtonContainer.innerHTML = `
            <button 
                id="takePhotoBtn" 
                class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition duration-200 mr-2"
            >
                Fotoğraf Çek
            </button>
            <button 
                id="uploadPhotoBtn" 
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition duration-200"
            >
                Fotoğraf Yükle
            </button>
        `;
        
        setTimeout(() => {
            const takePhotoBtn = document.getElementById('takePhotoBtn');
            const uploadPhotoBtn = document.getElementById('uploadPhotoBtn');
            
            if (takePhotoBtn) {
                takePhotoBtn.onclick = () => {
                    alert("Sprint 3'te kamera açılacak");
                };
            }
            
            if (uploadPhotoBtn) {
                uploadPhotoBtn.onclick = () => {
                    closeWizardModal();
                    if (photoInput) {
                        photoInput.focus();
                        photoInput.click();
                    }
                };
            }
        }, 0);
    } else {
        nextBtn.classList.remove('hidden');
        if (!nextBtn.parentElement) {
            nextButtonContainer.innerHTML = '';
            nextButtonContainer.appendChild(nextBtn);
        }
    }
}

function goToPrevWizardStep() {
    if (currentWizardStep > 0) {
        currentWizardStep--;
        renderWizardStep();
        renderWizardStepIndicator();
    }
}

function goToNextWizardStep() {
    if (currentWizardStep < wizardSteps.length - 1) {
        currentWizardStep++;
        renderWizardStep();
        renderWizardStepIndicator();
    }
}

// ============================================================================
// Upload Handler
// ============================================================================
async function handleUpload(event) {
    event.preventDefault();
    
    if (!photoInput || !photoInput.files || photoInput.files.length === 0) {
        alert('Lütfen bir fotoğraf seçin');
        return;
    }
    
    const file = photoInput.files[0];
    const formData = new FormData();
    formData.append('photo', file);
    
    // Create preview URL
    if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
    }
    currentImageUrl = URL.createObjectURL(file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            alert(data.error || 'Fotoğraf yüklenirken bir hata oluştu');
            return;
        }
        
        currentJobId = data.job_id;
        
        // Use preview_url from backend if available, otherwise use blob URL
        const previewUrl = data.preview_url || currentImageUrl;
        currentPreviewUrl = previewUrl;
        
        // Reset state
        processingStart = null;
        backendDonePayload = null;
        
        // Start polling (it will call startProcessingUI)
        startPolling();
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Fotoğraf yüklenirken bir hata oluştu');
    }
}

// ============================================================================
// Processing Modal Functions
// ============================================================================
function startProcessingUI(previewUrl) {
    if (!processingModal) return;
    
    // Reset step states
    stepStates = {};
    processingSteps.forEach(step => {
        stepStates[step.key] = "pending";
    });
    
    // Open modal
    processingModal.classList.remove('hidden');
    
    // Set preview
    setPreview(previewUrl);
    
    // Overlay kesin aktif
    setOverlayMode("processing");
    
    // Start scan loop
    startScanLoop();
    
    // Start processing timer
    processingStart = Date.now();
    
    // UI timer: her 100ms checklist'i güncelle
    if (uiTimer) clearInterval(uiTimer);
    uiTimer = setInterval(() => {
        const elapsed = Date.now() - processingStart;
        updateChecklistByElapsed(elapsed);
    }, 100);
    
    renderProcessingSteps();
    
    return processingStart;
}

function openProcessingModal() {
    // Legacy function - use startProcessingUI instead
    if (currentPreviewUrl) {
        return startProcessingUI(currentPreviewUrl);
    }
}

function closeProcessingModal() {
    if (!processingModal) return;
    if (uiTimer) {
        clearInterval(uiTimer);
        uiTimer = null;
    }
    stopScanLoop();
    processingModal.classList.add('hidden');
}

function renderProcessingSteps() {
    const progressSteps = document.getElementById('progressSteps');
    if (!progressSteps) return;
    
    progressSteps.innerHTML = processingSteps.map((step) => {
        const state = stepStates[step.key] || "pending";
        const isDone = state === "done";
        const isProcessing = state === "processing";
        
        return `
            <div class="flex items-center gap-4">
                <div class="flex-shrink-0">
                    ${isDone ? `
                        <div class="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                        </div>
                    ` : isProcessing ? `
                        <div class="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        </div>
                    ` : `
                        <div class="w-8 h-8 rounded-full bg-gray-300"></div>
                    `}
                </div>
                <div class="flex-1">
                    <p class="font-semibold text-gray-800 ${isProcessing ? 'text-blue-600' : isDone ? 'text-green-600' : 'text-gray-400'}">
                        ${step.name}
                    </p>
                </div>
            </div>
        `;
    }).join('');
}

function markStepDone(stepKey) {
    stepStates[stepKey] = "done";
    renderProcessingSteps();
}

function markStepPending(stepKey) {
    stepStates[stepKey] = "pending";
    renderProcessingSteps();
}

function markStepProcessing(stepKey) {
    stepStates[stepKey] = "processing";
    renderProcessingSteps();
}

function updateChecklistByElapsed(elapsed) {
    for (const s of STEP_POINTS) {
        if (elapsed >= s.t) {
            markStepDone(s.key);
        } else {
            markStepPending(s.key);
        }
    }
}

// ============================================================================
// Preview Functions
// ============================================================================
function setPreview(previewUrl) {
    const previewImage = document.getElementById('previewImage');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    
    if (previewUrl && previewImage && previewPlaceholder) {
        previewImage.src = previewUrl;
        previewImage.classList.remove('hidden');
        previewPlaceholder.classList.add('hidden');
    }
}

function setOverlayMode(mode) {
    const aiOverlay = document.getElementById('aiOverlay');
    if (!aiOverlay) return;
    
    if (mode === "processing") {
        aiOverlay.classList.add('aiOverlay--active');
        aiOverlay.classList.remove('aiOverlay--paused');
    } else if (mode === "done") {
        aiOverlay.classList.add('aiOverlay--paused');
        aiOverlay.classList.remove('aiOverlay--active');
    }
}

// Backward compatibility
function setProcessingMode(isProcessing) {
    setOverlayMode(isProcessing ? "processing" : "done");
}

// ============================================================================
// Scan Loop Functions
// ============================================================================
function startScanLoop() {
    const band = document.getElementById("scanBand");
    const stage = document.querySelector(".previewStage");
    if (!band || !stage) return;
    
    const h = stage.getBoundingClientRect().height;
    const startY = -30;
    const endY = h + 30;
    const speedPxPerSec = 220; // yavaş - gerçekçi
    let y = startY;
    let last = performance.now();
    
    scanRunning = true;
    band.classList.remove("fadeOut");
    
    function tick(now) {
        if (!scanRunning) return;
        const dt = (now - last) / 1000;
        last = now;
        
        y += speedPxPerSec * dt;
        if (y > endY) y = startY;
        
        band.style.transform = `translateY(${y}px)`;
        scanRaf = requestAnimationFrame(tick);
    }
    scanRaf = requestAnimationFrame(tick);
}

function stopScanLoop() {
    scanRunning = false;
    if (scanRaf) {
        cancelAnimationFrame(scanRaf);
        scanRaf = null;
    }
    const band = document.getElementById("scanBand");
    if (band) {
        band.classList.add("fadeOut"); // 0.35s fade
    }
}

function startProcessingAnimation() {
    processingStepIndex = 0;
    renderProcessingSteps();
    
    processingSteps.forEach((step, index) => {
        if (index < processingSteps.length - 1) {
            setTimeout(() => {
                processingStepIndex = index + 1;
                renderProcessingSteps();
            }, step.duration);
        }
    });
}

function stopProcessingAnimation() {
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
}

// ============================================================================
// Polling
// ============================================================================
function startPolling() {
    if (!currentJobId) return;
    
    // Processing UI'yi başlat (eğer başlatılmadıysa)
    if (!processingStart) {
        processingStart = startProcessingUI(currentPreviewUrl || currentImageUrl);
    }
    
    // Polling interval
    pollingInterval = setInterval(async () => {
        await pollJob(currentJobId);
    }, 900);
    
    // İlk kontrol hemen yap
    pollJob(currentJobId);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function pollJob(jobId) {
    if (!jobId) {
        stopPolling();
        return;
    }
    
    try {
        const response = await fetch(`/job/${jobId}/status`);
        const data = await response.json();
        
        // preview_url data içinde varsa set et (gecikmeli gelirse)
        if (data.preview_url) {
            currentPreviewUrl = data.preview_url;
            setPreview(data.preview_url);
        }
        
        // Processing aşamasında sweep her zaman aktif kalsın
        setOverlayMode("processing");
        
        if (data.status === "done") {
            backendDonePayload = data;
            
            // Interval'ı durdur
            stopPolling();
            
            const finish = () => {
                if (uiTimer) {
                    clearInterval(uiTimer);
                    uiTimer = null;
                }
                stopScanLoop(); // Aynı anda biter
                showResultScreen(backendDonePayload);
            };
            
            const elapsed = Date.now() - processingStart;
            const waitMore = Math.max(0, MIN_PROCESSING_MS - elapsed);
            setTimeout(finish, waitMore);
            
        } else if (data.status === "not_found") {
            stopPolling();
            if (uiTimer) {
                clearInterval(uiTimer);
                uiTimer = null;
            }
            stopScanLoop();
            setOverlayMode("done");
            alert('Job bulunamadı');
        }
    } catch (error) {
        console.error('Polling error:', error);
    }
}

// Legacy function - backward compatibility
async function checkJobStatus() {
    await pollJob(currentJobId);
}

// ============================================================================
// Result Modal Functions
// ============================================================================
function showResultScreen(jobData) {
    if (!resultModal) return;
    
    // Processing modal'ı kapat
    closeProcessingModal();
    
    // Done ekranında overlay pause kalmalı (scan görünmesin)
    setOverlayMode("done");
    
    // Checklist render et
    renderChecklist(jobData.checks || {});
    renderResultPreview(jobData.overlay || {});
    updateResultTitle(jobData.result, jobData.reasons || []);
    
    resultModal.classList.remove('hidden');
}

// Legacy function - backward compatibility
function openResultModal(jobData) {
    showResultScreen(jobData);
}

function closeResultModal() {
    if (!resultModal) return;
    resultModal.classList.add('hidden');
}

function renderChecklist(checks) {
    const checklistContainer = document.getElementById('checklistContainer');
    if (!checklistContainer) return;
    
    const checklistHtml = checklistItems.map(item => {
        const checkValue = checks[item.key];
        const isPass = checkValue === true;
        const isFail = checkValue === false;
        const isUnknown = checkValue === null || checkValue === undefined;
        
        return `
            <div class="flex items-center gap-4 p-4 rounded-lg ${isFail ? 'bg-red-50 border border-red-200' : isPass ? 'bg-green-50 border border-green-200' : 'bg-gray-50 border border-gray-200'}">
                <div class="flex-shrink-0">
                    ${isPass ? `
                        <div class="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
                            <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                        </div>
                    ` : isFail ? `
                        <div class="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center">
                            <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </div>
                    ` : `
                        <div class="w-6 h-6 rounded-full bg-gray-300"></div>
                    `}
                </div>
                <div class="flex-1">
                    <p class="font-semibold ${isFail ? 'text-red-700' : isPass ? 'text-green-700' : 'text-gray-500'}">
                        ${item.label}
                    </p>
                    ${isPass ? '<p class="text-sm text-green-600 mt-1">Geçti</p>' : isFail ? '<p class="text-sm text-red-600 mt-1">Geçmedi</p>' : ''}
                </div>
            </div>
        `;
    }).join('');
    
    checklistContainer.innerHTML = checklistHtml;
}

function renderResultPreview(overlay) {
    const resultPreviewImage = document.getElementById('resultPreviewImage');
    const resultPreviewPlaceholder = document.getElementById('resultPreviewPlaceholder');
    
    // Use preview_url from overlay if available, otherwise use currentImageUrl
    const previewUrl = overlay.preview_url || currentImageUrl;
    
    if (previewUrl && resultPreviewImage && resultPreviewPlaceholder) {
        resultPreviewImage.src = previewUrl;
        resultPreviewImage.classList.remove('hidden');
        resultPreviewPlaceholder.classList.add('hidden');
    }
}

function updateResultTitle(result, reasons) {
    const resultTitle = document.getElementById('resultTitle');
    const retakePhotoBtn = document.getElementById('retakePhotoBtn');
    const continueBtn = document.getElementById('continueBtn');
    const checklistContainer = document.getElementById('checklistContainer');
    
    if (resultTitle) {
        resultTitle.textContent = result === 'pass' ? 'İlk kontrolden geçildi' : 'Fotoğraf uygun değil';
    }
    
    if (result === 'fail' && retakePhotoBtn) {
        retakePhotoBtn.classList.remove('bg-gray-200', 'hover:bg-gray-300', 'text-gray-700');
        retakePhotoBtn.classList.add('bg-red-600', 'hover:bg-red-700', 'text-white');
    }
    
    // Add reasons list below checklist if fail
    if (result === 'fail' && reasons.length > 0 && checklistContainer) {
        const reasonsHtml = `
            <div class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p class="font-semibold text-red-700 mb-2">Sorunlar:</p>
                <ul class="list-disc list-inside space-y-1">
                    ${reasons.map(reason => `<li class="text-sm text-red-600">${reason}</li>`).join('')}
                </ul>
            </div>
        `;
        checklistContainer.insertAdjacentHTML('beforeend', reasonsHtml);
    }
}

// ============================================================================
// Event Listeners
// ============================================================================
if (uploadForm) {
    uploadForm.addEventListener('submit', handleUpload);
}

// Wizard modal
const openWizardBtn = document.getElementById('openWizardBtn');
const closeWizardBtn = document.getElementById('closeWizardBtn');
const wizardOverlay = document.getElementById('wizardOverlay');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

if (openWizardBtn) {
    openWizardBtn.addEventListener('click', openWizardModal);
}
if (closeWizardBtn) {
    closeWizardBtn.addEventListener('click', closeWizardModal);
}
if (wizardOverlay) {
    wizardOverlay.addEventListener('click', closeWizardModal);
}
if (prevBtn) {
    prevBtn.addEventListener('click', goToPrevWizardStep);
}
if (nextBtn) {
    nextBtn.addEventListener('click', goToNextWizardStep);
}

// Result modal
const resultOverlay = document.getElementById('resultOverlay');
const retakePhotoBtn = document.getElementById('retakePhotoBtn');
const continueBtn = document.getElementById('continueBtn');

if (resultOverlay) {
    resultOverlay.addEventListener('click', closeResultModal);
}
if (retakePhotoBtn) {
    retakePhotoBtn.addEventListener('click', () => {
        closeResultModal();
        if (currentImageUrl) {
            URL.revokeObjectURL(currentImageUrl);
            currentImageUrl = null;
        }
        if (photoInput) {
            photoInput.value = '';
        }
        // Redirect to home page
        window.location.href = '/';
    });
}

// Close modals on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (wizardModal && !wizardModal.classList.contains('hidden')) {
            closeWizardModal();
        } else if (processingModal && !processingModal.classList.contains('hidden')) {
            // Don't allow closing processing modal
        } else if (resultModal && !resultModal.classList.contains('hidden')) {
            closeResultModal();
        }
    }
});


