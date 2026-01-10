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
// Processing Steps & Timing (Sprint 2)
// ============================================================================
const MIN_PROCESSING_MS = 1500; // Minimum processing süresi (1.5 saniye - faster UX)
const MAX_PROCESSING_MS = 8000; // Maximum processing süresi (8 saniye)
const SCAN_DURATION_MS = 2500; // Scan animasyonu süresi (2.5 saniye - faster UX)
const STEP_POINTS = [
    { key: "crop", t: 400 },       // 0.4s - faster
    { key: "bg_remove", t: 800 },  // 0.8s - faster
    { key: "resize", t: 1200 },    // 1.2s - faster
    { key: "analyze", t: 1800 }    // 1.8s - finishes before scan ends
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
let serverPreviewUrl = null;  // Supabase signed URL (for processed result)
let processingStepIndex = 0;
let processingInterval = null;
let pollingInterval = null;
let pollingDelay = 1000;  // Start at 1s
let pollingAttempts = 0;
const POLLING_DELAYS = [1000, 2000, 3000, 5000, 8000];  // Exponential backoff
let processingStart = 0;
let backendDonePayload = null;
let stepStates = {}; // { crop: "pending|processing|done", ... }
let scanRunning = false;
let scanRaf = null;
let uiTimer = null;
let acknowledgedIssueIds = []; // Track acknowledged warning issues for current session
let isPhotoProcessed = false; // Track if photo has been processed (for download button)
let processedPhotoUrl = null; // Store the processed photo URL

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
        
        let data;
        try {
            data = await response.json();
        } catch (e) {
            console.error('JSON parse error:', e);
            alert('Sunucu yanıtı okunamadı. Lütfen tekrar deneyin.');
            return;
        }
        
        if (!response.ok || !data.success) {
            console.error('Upload error:', data);
            alert(data.error || 'Fotoğraf yüklenirken bir hata oluştu');
            return;
        }
        
        currentJobId = data.job_id;
        
        // ALWAYS use local blob URL for fast preview during processing
        // Server URL (Supabase signed URL) is slow and unnecessary for preview
        currentPreviewUrl = currentImageUrl;  // Use local blob, not server URL
        
        // Store server preview URL for later (if needed)
        serverPreviewUrl = data.preview_url;
        
        // Reset state
        processingStart = null;
        backendDonePayload = null;
        
        // Start polling (it will call startProcessingUI)
        startPolling();
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Fotoğraf yüklenirken bir hata oluştu: ' + error.message);
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
// Scan Loop Functions (Sprint 2 - 7-8 saniye, no loop)
// ============================================================================
let scanStartTime = 0;
let scanRafId = null;

function startScanLoop() {
    const band = document.getElementById("scanBand");
    const stage = document.querySelector(".previewStage");
    if (!band || !stage) return;
    
    // Remove fadeOut class if present
    band.classList.remove("fadeOut");
    
    scanStartTime = Date.now();
    const stageHeight = stage.getBoundingClientRect().height;
    const startY = -10;
    const endY = stageHeight + 10;
    
    function animate() {
        const elapsed = Date.now() - scanStartTime;
        const progress = Math.min(1.0, elapsed / SCAN_DURATION_MS);
        
        // Calculate Y position (startY to endY)
        const yPos = startY + (endY - startY) * progress;
        band.style.transform = `translateY(${yPos}px)`;
        
        // Stop when scan is complete (no loop)
        if (progress >= 1.0) {
            stopScanLoop();
            return;
        }
        
        scanRafId = requestAnimationFrame(animate);
    }
    
    scanRafId = requestAnimationFrame(animate);
}

function stopScanLoop() {
    if (scanRafId) {
        cancelAnimationFrame(scanRafId);
        scanRafId = null;
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
    
    // Reset polling state
    pollingAttempts = 0;
    pollingDelay = POLLING_DELAYS[0];
    
    // İlk kontrol hemen yap, sonra exponential backoff ile devam et
    pollJob(currentJobId);
}

function scheduleNextPoll(jobId) {
    // Calculate next delay with exponential backoff
    const delayIndex = Math.min(pollingAttempts, POLLING_DELAYS.length - 1);
    pollingDelay = POLLING_DELAYS[delayIndex];
    pollingAttempts++;
    
    console.log(`[POLLING] Next poll in ${pollingDelay}ms (attempt ${pollingAttempts})`);
    
    pollingInterval = setTimeout(async () => {
        await pollJob(jobId);
    }, pollingDelay);
}

function stopPolling() {
    if (pollingInterval) {
        clearTimeout(pollingInterval);
        pollingInterval = null;
    }
    pollingAttempts = 0;
    pollingDelay = POLLING_DELAYS[0];
}

async function pollJob(jobId) {
    if (!jobId) {
        stopPolling();
        return;
    }
    
    try {
        const response = await fetch(`/job/${jobId}/status`);
        
        // Handle rate limiting (429)
        if (response.status === 429) {
            const retryAfter = response.headers.get('Retry-After') || 2;
            console.log(`[POLLING] Rate limited. Retry after ${retryAfter}s`);
            setTimeout(() => scheduleNextPoll(jobId), retryAfter * 1000);
            return;
        }
        
        const data = await response.json();
        
        // #region agent log - Full JSON response logging
        console.log('=== pollJob FULL JSON RESPONSE ===');
        console.log(JSON.stringify(data, null, 2));
        console.log('  data.issues:', data.issues);
        console.log('  data.can_continue:', data.can_continue);
        console.log('  data.pending_ack_ids:', data.pending_ack_ids);
        console.log('  data.final_status:', data.final_status);
        // #endregion
        
        // Debug logging
        console.log('[DEBUG] pollJob response:', {
            status: data.status,
            result: data.result,
            normalized_url: data.normalized_url,
            preview_url: data.preview_url,
            ok: data.ok
        });
        
        // preview_url data içinde varsa set et (gecikmeli gelirse)
        if (data.preview_url) {
            currentPreviewUrl = data.preview_url;
            setPreview(data.preview_url);
        }
        
        // Processing aşamasında sweep her zaman aktif kalsın
        setOverlayMode("processing");
        
        // Handle queued status - reset backoff when queued
        if (data.status === "queued") {
            console.log(`[POLLING] Job queued at position ${data.queue_position}`);
            // Keep polling but don't show error
            scheduleNextPoll(jobId);
            return;
        }
        
        // Handle processing status - continue polling
        if (data.status === "processing") {
            scheduleNextPoll(jobId);
            return;
        }
        
        if (data.status === "done") {
            backendDonePayload = data;
            
            // Interval'ı durdur
            stopPolling();
            
            // Log timing info if available
            if (data.timing) {
                console.log(`[TIMING] download=${data.timing.download_ms}ms, analyze=${data.timing.analyze_ms}ms, db=${data.timing.db_ms}ms, total=${data.timing.total_ms}ms`);
            }
            
            // Log warning if DB save failed (but don't block user)
            if (data.db_saved === false) {
                console.warn('[WARNING] Job completed but DB save failed:', data.db_error);
                console.warn('[WARNING] User can still download/preview. Retries may be happening server-side.');
            }
            
            const finish = () => {
                if (uiTimer) {
                    clearInterval(uiTimer);
                    uiTimer = null;
                }
                stopScanLoop(); // Aynı anda biter
                showResultScreen(backendDonePayload);
            };
            
            const elapsed = Date.now() - processingStart;
            // Clamp processing time: min 5s, max 12s
            const clampedElapsed = Math.max(MIN_PROCESSING_MS, Math.min(MAX_PROCESSING_MS, elapsed));
            const waitMore = Math.max(0, clampedElapsed - elapsed);
            
            // Ensure scan completes before showing result
            const scanElapsed = Date.now() - scanStartTime;
            const scanWait = Math.max(0, SCAN_DURATION_MS - scanElapsed);
            const totalWait = Math.max(waitMore, scanWait);
            
            setTimeout(finish, totalWait);
            
        } else if (data.status === "not_found") {
            stopPolling();
            if (uiTimer) {
                clearInterval(uiTimer);
                uiTimer = null;
            }
            stopScanLoop();
            setOverlayMode("done");
            alert('Job bulunamadı');
        } else if (data.status === "FAIL" || data.status === "failed") {
            // Job failed - stop polling and show error
            stopPolling();
            if (uiTimer) {
                clearInterval(uiTimer);
                uiTimer = null;
            }
            stopScanLoop();
            backendDonePayload = data;
            showResultScreen(data);
        } else if (data.status === "error") {
            // Server error - schedule retry with backoff
            console.log('[POLLING] Server error, retrying...');
            scheduleNextPoll(jobId);
        } else {
            // Unknown status - continue polling
            scheduleNextPoll(jobId);
        }
    } catch (error) {
        console.error('Polling error:', error);
        // Network error - schedule retry with backoff
        scheduleNextPoll(jobId);
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
    console.log('=== showResultScreen DEBUG ===');
    console.log('  jobData:', jobData);
    console.log('  jobData.final_status:', jobData?.final_status);
    console.log('  jobData.can_continue:', jobData?.can_continue);
    console.log('  jobData.issues:', jobData?.issues);
    console.log('  jobData.pending_ack_ids:', jobData?.pending_ack_ids);
    
    // EYEWEAR DEBUG - Backend'den gelen metrikleri göster
    console.log('=== EYEWEAR DEBUG ===');
    console.log('  eyewear_type:', jobData?.metrics?.eyewear_type);
    console.log('  eyewear_glasses_score:', jobData?.metrics?.eyewear_glasses_score);
    console.log('  eyewear_sunglasses_score:', jobData?.metrics?.eyewear_sunglasses_score);
    console.log('  frame_presence_score:', jobData?.metrics?.frame_presence_score);
    console.log('  glasses_score (raw):', jobData?.metrics?.glasses_score);
    console.log('  sunglasses_score (raw):', jobData?.metrics?.sunglasses_score);
    console.log('  edge_density:', jobData?.metrics?.edge_density);
    console.log('  highlight_ratio:', jobData?.metrics?.highlight_ratio);
    
    if (!resultModal) {
        console.log('  ERROR: resultModal not found!');
        return;
    }
    
    // Processing modal'ı kapat
    closeProcessingModal();
    
    // Done ekranında overlay pause kalmalı (scan görünmesin)
    setOverlayMode("done");
    
    // Reset state for new photo session
    acknowledgedIssueIds = [];
    isPhotoProcessed = false;
    processedPhotoUrl = null;
    
    // Checklist render et
    renderChecklist(jobData.checks || {});
    renderResultPreview(jobData.overlay || {}, jobData);
    
    // Render issues with new format (PASS/WARN/FAIL, checkboxes for requires_ack)
    const issuesArray = jobData.issues || [];
    
    // #region agent log - Derived values from backend response
    console.log('=== showResultScreen DERIVED VALUES ===');
    console.log('  issues:', issuesArray);
    console.log('  can_continue (from backend):', jobData.can_continue);
    console.log('  pending_ack_ids (from backend):', jobData.pending_ack_ids);
    console.log('  final_status (from backend):', jobData.final_status);
    
    // Derive from issues
    const hasFail = issuesArray.some(i => i.severity === 'FAIL');
    const hasWarn = issuesArray.some(i => i.severity === 'WARN');
    const requiredAckIds = issuesArray
        .filter(i => i.severity === 'WARN' && i.requires_ack === true)
        .map(i => i.id);
    console.log('  Derived: hasFail=', hasFail, 'hasWarn=', hasWarn, 'requiredAckIds=', requiredAckIds);
    // #endregion
    
    console.log('  Calling renderIssuesV2 with:', issuesArray);
    console.log('  issuesArray type:', Array.isArray(issuesArray));
    console.log('  issuesArray[0]:', issuesArray[0]);
    console.log('  issuesArray[0]?.severity:', issuesArray[0]?.severity);
    renderIssuesV2(issuesArray, jobData.pending_ack_ids || []);
    
    // Update title based on issues array (derive from issues, not just final_status)
    updateResultTitle(jobData.final_status, issuesArray);
    
    // Debug panel render et (dev mode) - with eyewear info
    renderDebugPanel(jobData.metrics || {});
    
    // Hide continue button by default (we'll use auto-processing)
    const continueBtn = document.getElementById('continueBtn');
    if (continueBtn) {
        // Only show as "İndir" button when processing is complete
        if (isPhotoProcessed && processedPhotoUrl) {
            continueBtn.textContent = 'İndir';
            continueBtn.classList.remove('hidden', 'bg-gray-400', 'cursor-not-allowed');
            continueBtn.classList.add('bg-blue-600', 'hover:bg-blue-700', 'cursor-pointer');
            continueBtn.disabled = false;
        continueBtn.onclick = () => {
                window.location.href = processedPhotoUrl;
        };
        } else {
            // Hide button during analysis/before processing
            continueBtn.classList.add('hidden');
        }
    }
    
    resultModal.classList.remove('hidden');
    
    // Show validation result screen instead of auto-processing
    // This gives user control to proceed to checkout
    const previewUrl = jobData.preview_url || jobData.preview_image || currentPreviewUrl;
    setTimeout(() => {
        showValidationResultScreen(jobData, previewUrl);
    }, 100);
}

// ============================================================================
// Issue Rendering V2 (New Format with PASS/WARN/FAIL and checkboxes)
// ============================================================================
// BEHAVIOR:
// - FAIL (red): X icon, Continue DISABLED, no override
// - WARN (yellow): Warning icon, checkbox if requires_ack=true
// - PASS (green): Tick icon
// ============================================================================
function renderIssuesV2(issues, pendingAckIds) {
    console.log('=== renderIssuesV2 DEBUG ===');
    console.log('  issues:', issues);
    console.log('  pendingAckIds:', pendingAckIds);
    
    const checklistContainer = document.getElementById('checklistContainer');
    if (!checklistContainer) {
        console.log('  ERROR: checklistContainer not found!');
        return;
    }
    
    // Remove any existing issues section
    const existingIssues = checklistContainer.querySelector('.issues-section');
    if (existingIssues) {
        existingIssues.remove();
    }
    
    if (!issues || issues.length === 0) {
        console.log('  No issues to render, returning early');
        return;
    }
    
    console.log('  Rendering', issues.length, 'issues');
    
    // Separate by severity (use uppercase from new format)
    const failIssues = issues.filter(i => i.severity === 'FAIL');
    const warnIssues = issues.filter(i => i.severity === 'WARN');
    const passIssues = issues.filter(i => i.severity === 'PASS');
    
    let issuesHtml = '<div class="issues-section mt-6 space-y-4">';
    
    // Render FAIL issues (blocking - RED, X icon)
    failIssues.forEach(issue => {
        issuesHtml += `
            <div class="p-4 bg-red-50 border-2 border-red-400 rounded-lg">
                <div class="flex items-start gap-3">
                    <div class="flex-shrink-0 mt-0.5">
                        <svg class="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <div>
                        <p class="font-bold text-red-700">${issue.title_tr || issue.id}</p>
                        <p class="text-sm text-red-600 mt-1">${issue.message_tr}</p>
                    </div>
                </div>
            </div>
        `;
    });
    
    // Render WARN issues (yellow, warning icon, optional checkbox)
    warnIssues.forEach(issue => {
        const needsAck = issue.requires_ack === true;
        const isAcked = acknowledgedIssueIds.includes(issue.id);
        
        issuesHtml += `
            <div class="p-4 bg-yellow-50 border-2 border-yellow-400 rounded-lg" data-issue-id="${issue.id}">
                <div class="flex items-start gap-3">
                    <div class="flex-shrink-0 mt-0.5">
                        <svg class="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                        </svg>
                    </div>
                    <div class="flex-1">
                        <p class="font-bold text-yellow-700">${issue.title_tr || issue.id}</p>
                        <p class="text-sm text-yellow-700 mt-1">${issue.message_tr}</p>
                        ${needsAck ? `
                            <label class="flex items-start gap-2 mt-3 cursor-pointer select-none">
                                <input type="checkbox" 
                                       class="ack-checkbox mt-0.5 w-4 h-4 text-yellow-600 border-yellow-400 rounded focus:ring-yellow-500"
                                       data-issue-id="${issue.id}"
                                       ${isAcked ? 'checked' : ''}
                                       onchange="handleAckCheckbox(this, '${issue.id}')">
                                <span class="text-sm text-yellow-800">
                                    Gözlük takıyorum. Bazı ülkelerde kabul edilmeyebilir. Riski anladım ve devam etmek istiyorum.
                                </span>
                            </label>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    });
    
    // Render PASS issues (green, tick icon) - if any explicit pass issues
    passIssues.forEach(issue => {
        issuesHtml += `
            <div class="p-4 bg-green-50 border-2 border-green-400 rounded-lg">
                <div class="flex items-start gap-3">
                    <div class="flex-shrink-0 mt-0.5">
                        <svg class="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <div>
                        <p class="font-bold text-green-700">${issue.title_tr || issue.id}</p>
                        <p class="text-sm text-green-600 mt-1">${issue.message_tr}</p>
                    </div>
                </div>
            </div>
        `;
    });
    
    // Add disclaimer at the bottom
    if (warnIssues.length > 0 || failIssues.length > 0) {
        issuesHtml += `
            <div class="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <p class="text-xs text-gray-500 text-center">
                    Bu kontrol otomatik analizdir. Nihai kabul ilgili ülke/kuruma aittir.
                </p>
            </div>
        `;
    }
    
    issuesHtml += '</div>';
    
    checklistContainer.insertAdjacentHTML('beforeend', issuesHtml);
}

// Handle acknowledgement checkbox changes
function handleAckCheckbox(checkbox, issueId) {
    console.log('=== handleAckCheckbox ===', issueId, checkbox.checked);
    
    if (checkbox.checked) {
        // Add to acknowledged list if not already there
        if (!acknowledgedIssueIds.includes(issueId)) {
            acknowledgedIssueIds.push(issueId);
        }
    } else {
        // Remove from acknowledged list
        acknowledgedIssueIds = acknowledgedIssueIds.filter(id => id !== issueId);
    }
    
    console.log('  acknowledgedIssueIds:', acknowledgedIssueIds);
    
    // Check if all required acks are done - if so, auto-start processing
    checkAndAutoProcess();
}

// Check if we can auto-process and start if ready
function checkAndAutoProcess() {
    if (!currentJobId || !backendDonePayload) return;
    
    const issues = backendDonePayload.issues || [];
    const hasFailIssues = issues.some(i => i.severity === 'FAIL');
    
    // If there are FAIL issues, don't auto-process
    if (hasFailIssues) return;
    
    // Check if all required acknowledgements are done
    const requiredAckIds = issues
        .filter(i => i.severity === 'WARN' && i.requires_ack === true)
        .map(i => i.id);
    
    const allAcksCompleted = requiredAckIds.every(id => acknowledgedIssueIds.includes(id));
    
    console.log('=== checkAndAutoProcess ===');
    console.log('  hasFailIssues:', hasFailIssues);
    console.log('  requiredAckIds:', requiredAckIds);
    console.log('  acknowledgedIssueIds:', acknowledgedIssueIds);
    console.log('  allAcksCompleted:', allAcksCompleted);
    
    if (allAcksCompleted) {
        console.log('  ✅ All conditions met - auto-starting processing!');
        startAutoProcessing();
    }
}

// Auto-start processing (no button click needed)
function startAutoProcessing() {
    if (!currentJobId || isPhotoProcessed) return;
    
    console.log('🚀 [AUTO-PROCESS] Starting PhotoRoom processing for job:', currentJobId);
    
    // Show processing state in UI
    showProcessingState();
    
    // Start processing
    processPhoto(currentJobId, acknowledgedIssueIds);
}

// Show processing state in the result modal
function showProcessingState() {
    const resultTitle = document.getElementById('resultTitle');
    const continueBtn = document.getElementById('continueBtn');
    const retakePhotoBtn = document.getElementById('retakePhotoBtn');
    
    if (resultTitle) {
        resultTitle.innerHTML = 'Fotoğraf işleniyor... <span class="inline-block animate-spin">⏳</span>';
        resultTitle.className = 'text-2xl font-bold text-blue-600';
    }
    
    // Hide buttons during processing
    if (continueBtn) {
        continueBtn.classList.add('hidden');
    }
    if (retakePhotoBtn) {
        retakePhotoBtn.classList.add('hidden');
    }
}

// Update result title based on issues array (derive from issues, not just final_status)
function updateResultTitle(finalStatus, issues) {
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');
    const retakePhotoBtn = document.getElementById('retakePhotoBtn');
    
    if (!resultTitle) return;
    
    // #region agent log - Derive status from issues
    console.log('=== updateResultTitle DEBUG ===');
    console.log('  issues array:', issues);
    console.log('  issues length:', issues?.length);
    console.log('  issues[0]:', issues?.[0]);
    console.log('  issues[0]?.severity:', issues?.[0]?.severity);
    console.log('  issues[0] keys:', issues?.[0] ? Object.keys(issues[0]) : 'N/A');
    
    // Derive status from issues array (single source of truth)
    const hasFailIssues = issues.some(i => i.severity === 'FAIL');
    const hasWarnIssues = issues.some(i => i.severity === 'WARN');
    
    console.log('  Derived: hasFailIssues=', hasFailIssues, 'hasWarnIssues=', hasWarnIssues);
    console.log('  Backend final_status:', finalStatus);
    // #endregion
    
    // Header rules (PhotoAID style):
    // - any FAIL => "Fotoğraf uygun değil" (red)
    // - else if any WARN => "İlk kontrolden geçildi" (yellow) - shows warning but can continue
    // - else => "İlk kontrolden geçildi" (green)
    if (hasFailIssues) {
        resultTitle.textContent = 'Fotoğraf uygun değil';
        resultTitle.className = 'text-3xl font-bold text-red-600';
        if (resultSubtitle) {
            resultSubtitle.textContent = 'Lütfen aşağıdaki sorunları düzeltin ve tekrar deneyin.';
            resultSubtitle.className = 'text-gray-600 mb-6';
        }
        console.log('  -> Set FAIL title (red)');
    } else if (hasWarnIssues) {
        resultTitle.textContent = 'İlk kontrolden geçildi';
        resultTitle.className = 'text-3xl font-bold text-gray-800';
        if (resultSubtitle) {
            resultSubtitle.textContent = 'Fotoğrafınız işleniyor...';
            resultSubtitle.className = 'text-gray-600 mb-6';
        }
        console.log('  -> Set WARN title (processing)');
    } else {
        resultTitle.textContent = 'İlk kontrolden geçildi';
        resultTitle.className = 'text-3xl font-bold text-gray-800';
        if (resultSubtitle) {
            resultSubtitle.textContent = 'Fotoğrafınız işleniyor...';
            resultSubtitle.className = 'text-gray-600 mb-6';
        }
        console.log('  -> Set PASS title (processing)');
    }
    
    // Retake button styling
    if (retakePhotoBtn) {
        if (hasFailIssues) {
            retakePhotoBtn.classList.remove('bg-white', 'hover:bg-gray-100', 'text-gray-700', 'border-gray-300');
            retakePhotoBtn.classList.add('bg-red-600', 'hover:bg-red-700', 'text-white', 'border-red-600');
            retakePhotoBtn.textContent = 'Yeni fotoğraf çek';
        } else {
            retakePhotoBtn.classList.remove('bg-red-600', 'hover:bg-red-700', 'text-white', 'border-red-600');
            retakePhotoBtn.classList.add('bg-white', 'hover:bg-gray-100', 'text-gray-700', 'border-gray-300');
            retakePhotoBtn.textContent = 'Fotoğrafı tekrar çekin';
        }
    }
}

function updateContinueButtonState() {
    const continueBtn = document.getElementById('continueBtn');
    if (!continueBtn) return;
    
    // Get current job data from backendDonePayload
    const issues = backendDonePayload?.issues || [];
    const pendingAckIds = backendDonePayload?.pending_ack_ids || [];
    const serverCanContinue = backendDonePayload?.can_continue ?? true;
    
    // #region agent log - Derive button state from issues
    console.log('=== updateContinueButtonState DEBUG ===');
    console.log('  issues:', issues);
    console.log('  pendingAckIds:', pendingAckIds);
    console.log('  server can_continue:', serverCanContinue);
    console.log('  acknowledgedIssueIds:', acknowledgedIssueIds);
    // #endregion
    
    // LOGIC (derive ONLY from issues, not server can_continue):
    // - FAIL => Continue DISABLED (no override)
    // - WARN with requires_ack=true => Continue DISABLED until checkbox checked
    // - WARN without requires_ack => Continue ENABLED
    // - PASS => Continue ENABLED
    
    const hasFailIssues = issues.some(i => i.severity === 'FAIL');
    
    // Check if all required acknowledgements are done
    const requiredAckIds = issues
        .filter(i => i.severity === 'WARN' && i.requires_ack === true)
        .map(i => i.id);
    
    const missingAck = requiredAckIds.some(id => !acknowledgedIssueIds.includes(id));
    
    // Button disabled if: hasFail OR missing required ack
    const continueEnabled = !hasFailIssues && !missingAck;
    
    console.log('=== updateContinueButtonState ===');
    console.log('  hasFailIssues:', hasFailIssues);
    console.log('  requiredAckIds:', requiredAckIds);
    console.log('  acknowledgedIssueIds:', acknowledgedIssueIds);
    console.log('  missingAck:', missingAck);
    console.log('  continueEnabled (derived):', continueEnabled);
    console.log('  server can_continue:', serverCanContinue);
    
    // Update button state based on derived logic
    continueBtn.disabled = !continueEnabled;
    
    if (continueEnabled) {
        continueBtn.classList.remove('bg-gray-400', 'cursor-not-allowed');
        continueBtn.classList.add('bg-blue-600', 'hover:bg-blue-700', 'cursor-pointer');
    } else {
        continueBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700', 'cursor-pointer');
        continueBtn.classList.add('bg-gray-400', 'cursor-not-allowed');
    }
    
    // Show dev line with server can_continue (for debugging)
    const debugPanel = document.getElementById('debugPanel');
    if (debugPanel) {
        const existingDevLine = debugPanel.querySelector('.server-can-continue');
        if (existingDevLine) {
            existingDevLine.remove();
        }
        const devLine = document.createElement('div');
        devLine.className = 'server-can-continue text-xs text-gray-500 mt-2';
        devLine.textContent = `server can_continue: ${serverCanContinue}`;
        debugPanel.appendChild(devLine);
    }
}

function renderDebugPanel(metrics) {
    const debugPanel = document.getElementById('debugPanel');
    const debugContent = document.getElementById('debugContent');
    
    if (!debugPanel || !debugContent) return;
    
    // Check if dev mode (show debug panel)
    const isDevMode = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    
    if (!isDevMode || !metrics || Object.keys(metrics).length === 0) {
        debugPanel.classList.add('hidden');
        return;
    }
    
    debugPanel.classList.remove('hidden');
    
    // Build debug info
    const debugInfo = [];
    
    // Eyewear detection info (priority display)
    if (metrics.eyewear_type !== undefined) {
        const eyewearEmoji = metrics.eyewear_type === 'sunglasses' ? '🕶️' : 
                            metrics.eyewear_type === 'glasses' ? '👓' : '👁️';
        debugInfo.push(`<strong>${eyewearEmoji} eyewear: ${metrics.eyewear_type} (${(metrics.eyewear_confidence * 100).toFixed(0)}%)</strong>`);
        if (metrics.eyewear_sunglasses_score !== undefined) {
            debugInfo.push(`  sunglasses_score: ${metrics.eyewear_sunglasses_score}`);
        }
        if (metrics.eyewear_glasses_score !== undefined) {
            debugInfo.push(`  glasses_score: ${metrics.eyewear_glasses_score}`);
        }
        if (metrics.eyewear_iris_visibility !== undefined) {
            debugInfo.push(`  iris_visibility: ${metrics.eyewear_iris_visibility}`);
        }
    }
    
    if (metrics.face_count !== undefined) {
        debugInfo.push(`face_count: ${metrics.face_count}`);
    }
    if (metrics.brightness_mean !== undefined) {
        debugInfo.push(`brightness_mean: ${metrics.brightness_mean}`);
    }
    if (metrics.blur_score !== undefined) {
        debugInfo.push(`blur_score: ${metrics.blur_score}`);
    }
    if (metrics.resolution_w && metrics.resolution_h) {
        debugInfo.push(`resolution: ${metrics.resolution_w}x${metrics.resolution_h}`);
    }
    
    debugContent.innerHTML = debugInfo.length > 0 
        ? debugInfo.join('<br>')
        : 'No debug info available';
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

function renderResultPreview(overlay, data) {
    const resultPreviewImage = document.getElementById('resultPreviewImage');
    const resultPreviewPlaceholder = document.getElementById('resultPreviewPlaceholder');
    
    // Priority: data.preview_url > data.preview_image > overlay.preview_url > currentPreviewUrl > currentImageUrl
    const previewUrl = data?.preview_url || data?.preview_image || overlay?.preview_url || currentPreviewUrl || currentImageUrl;
    
    // Debug logging
    console.log('[DEBUG] renderResultPreview:');
    console.log('  - data.preview_url:', data?.preview_url);
    console.log('  - data.preview_image:', data?.preview_image);
    console.log('  - overlay.preview_url:', overlay?.preview_url);
    console.log('  - currentPreviewUrl:', currentPreviewUrl);
    console.log('  - currentImageUrl:', currentImageUrl);
    console.log('  - SELECTED previewUrl:', previewUrl);
    
    if (previewUrl && resultPreviewImage && resultPreviewPlaceholder) {
        // Add cache busting if not already present
        const finalUrl = previewUrl.includes('?') ? previewUrl : previewUrl + '?t=' + Date.now();
        console.log('  - FINAL URL with cache bust:', finalUrl);
        
        resultPreviewImage.src = finalUrl;
        resultPreviewImage.classList.remove('hidden');
        resultPreviewPlaceholder.classList.add('hidden');
        } else {
        console.log('  - ERROR: No preview URL found or elements missing!');
    }
}

// Legacy functions removed - using new updateResultTitle(finalStatus, issues) instead

// ============================================================================
// Photo Processing Functions
// ============================================================================
async function processPhoto(jobId, acknowledgedIds = []) {
    if (!jobId) return;
    
    try {
        // Build request body with acknowledged issue IDs
        const body = {
            acknowledged_issue_ids: acknowledgedIds || []
        };
        
        console.log('[AUDIT] Processing photo with acknowledged issues:', acknowledgedIds);
        
        const response = await fetch(`/process/${jobId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            alert(data.error || 'Fotoğraf işlenirken bir hata oluştu');
            return;
        }
        
        // Start polling for processing status
        startProcessingPolling(jobId);
        
    } catch (error) {
        console.error('Processing error:', error);
        alert('Fotoğraf işlenirken bir hata oluştu');
    }
}

function startProcessingPolling(jobId) {
    // Show processing modal again with scan
    if (processingModal) {
        processingModal.classList.remove('hidden');
        setOverlayMode("processing");
        startScanLoop();
    }
    
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/job/${jobId}/status`);
            const data = await response.json();
            
            console.log('[DEBUG] startProcessingPolling:', data.status, data.processed, data.processed_url);
            
            // Check if processing is complete - handle both old and new response formats
            const isDone = (data.status === 'done' && data.processed === true) || 
                           (data.processing_status === 'done');
            const imageUrl = data.processed_url || data.normalized_url || data.output_url;
            
            if (isDone && imageUrl) {
                clearInterval(pollInterval);
                stopScanLoop();
                setOverlayMode("done");
                
                // Show result with processed photo
                setTimeout(() => {
                    closeProcessingModal();
                    showProcessedResult(jobId, imageUrl);
                }, 500);
            } else if (data.processing_status === 'error' || data.photoroom_error) {
                clearInterval(pollInterval);
                stopScanLoop();
                setOverlayMode("done");
                alert(data.processing_error || data.photoroom_error || 'İşleme hatası');
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

// Payment state
let isPaid = false;
let stripeConfig = null;

// Load Stripe config on page load
async function loadStripeConfig() {
    try {
        const response = await fetch('/api/config/stripe');
        stripeConfig = await response.json();
        console.log('[STRIPE] Config loaded:', stripeConfig);
        console.log('[STRIPE] Payments enabled:', stripeConfig?.enabled);
    } catch (error) {
        console.error('[STRIPE] Failed to load config:', error);
        // Default to disabled when config fails to load
        stripeConfig = { enabled: false, prices: {} };
    }
}

// Check if payments are enabled
function isPaymentsEnabled() {
    return stripeConfig?.enabled === true;
}

// Check payment status for a job
async function checkPaymentStatus(jobId) {
    try {
        const response = await fetch(`/api/payment/status/${jobId}`);
        const data = await response.json();
        console.log('[PAYMENT] Status:', data);
        return data;
    } catch (error) {
        console.error('[PAYMENT] Failed to check status:', error);
        return { paid: false };
    }
}

// Initiate Stripe Checkout
async function initiateCheckout(jobId, packageType, shippingData = null) {
    console.log('[CHECKOUT] Initiating for:', jobId, packageType, shippingData ? 'with shipping' : 'digital only');
    
    try {
        const payload = { 
            job_id: jobId, 
            package_type: packageType 
        };
        
        // Add shipping data for digital_print orders
        if (packageType === 'digital_print' && shippingData) {
            payload.shipping = {
                first_name: shippingData.firstname,
                last_name: shippingData.lastname,
                address: shippingData.address,
                city: shippingData.city,
                postal_code: shippingData.postalcode,
                phone: shippingData.phone,
                email: shippingData.email
            };
        }
        
        const response = await fetch('/api/checkout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success && data.checkout_url) {
            // Redirect to Stripe Checkout
            window.location.href = data.checkout_url;
        } else {
            alert(data.error || 'Ödeme başlatılamadı');
        }
    } catch (error) {
        console.error('[CHECKOUT] Error:', error);
        alert('Bir hata oluştu. Lütfen tekrar deneyin.');
    }
}

// Checkout state
let selectedPlan = 'digital';
let customerEmail = '';
let customerInfo = {
    firstName: '',
    lastName: '',
    email: '',
    address: '',
    city: '',
    postalCode: ''
};

// ============================================================================
// Validation Result Screen (PhotoAid style "İlk kontrolden geçildi")
// ============================================================================
function showValidationResultScreen(jobData, previewUrl) {
    if (!resultModal) return;
    
    console.log('[DEBUG] showValidationResultScreen:', jobData);
    
    const modalContent = resultModal.querySelector('.relative.bg-white');
    if (!modalContent) return;
    
    // Get checks data
    const checks = jobData.checks || {};
    const metrics = jobData.metrics || {};
    
    // Build checklist HTML
    const checklistItems = [
        { key: 'face_detected', label: 'Yüz tanındı' },
        { key: 'single_face', label: 'Fotoğrafta yalnızca bir yüz olmalı' },
        { key: 'min_size', label: 'Minimum boyut' },
        { key: 'aspect_ratio_ok', label: 'Fotoğraf oranları doğru' }
    ];
    
    const checklistHtml = checklistItems.map(item => {
        const passed = checks[item.key] !== false;
        return `
            <div class="flex items-center justify-between py-4 border-b border-slate-100 last:border-0">
                <span class="text-slate-700 font-medium">${item.label}</span>
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${passed ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'}">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        ${passed 
                            ? '<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>'
                            : '<path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>'
                        }
                    </svg>
                    Geçti
                </span>
            </div>
        `;
    }).join('');
    
    // Check if there are any fail issues
    const issues = jobData.issues || [];
    const hasFail = issues.some(i => i.severity === 'FAIL');
    const hasWarn = issues.some(i => i.severity === 'WARN' && i.requires_ack);
    
    // Build issues HTML if any
    let issuesHtml = '';
    if (hasFail || hasWarn) {
        issuesHtml = issues.map(issue => {
            if (issue.severity === 'FAIL') {
                return `
                    <div class="p-4 bg-red-50 border border-red-200 rounded-xl mb-3">
                        <div class="flex items-start gap-3">
                            <svg class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                            </svg>
                            <div>
                                <p class="font-semibold text-red-800">${issue.title_tr}</p>
                                <p class="text-sm text-red-600 mt-1">${issue.message_tr}</p>
                            </div>
                        </div>
                    </div>
                `;
            } else if (issue.severity === 'WARN' && issue.requires_ack) {
                return `
                    <div class="p-4 bg-amber-50 border border-amber-200 rounded-xl mb-3">
                        <div class="flex items-start gap-3">
                            <svg class="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                            </svg>
                            <div class="flex-1">
                                <p class="font-semibold text-amber-800">${issue.title_tr}</p>
                                <p class="text-sm text-amber-700 mt-1">${issue.message_tr}</p>
                                <label class="flex items-start gap-2 mt-3 cursor-pointer">
                                    <input type="checkbox" class="validation-ack-checkbox mt-0.5 w-4 h-4 text-amber-600 rounded" data-issue-id="${issue.id}">
                                    <span class="text-sm text-amber-800">Riski anladım ve devam etmek istiyorum.</span>
                                </label>
                            </div>
                        </div>
                    </div>
                `;
            }
            return '';
        }).join('');
    }
    
    modalContent.innerHTML = `
        <!-- Close Button -->
        <button id="closeValidationBtn" class="absolute top-4 right-4 text-slate-400 hover:text-slate-600 z-10 p-2 rounded-full hover:bg-slate-100 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
        </button>
        
        <!-- Main Content: 2-Column Layout -->
        <div class="flex flex-col lg:flex-row min-h-[500px]">
            
            <!-- LEFT COLUMN: Validation Results -->
            <div class="flex-1 p-8 lg:p-12">
                <p class="text-sm text-slate-400 mb-2">Adım 1/3</p>
                <h1 class="text-3xl lg:text-4xl font-bold text-slate-900 mb-4">${hasFail ? 'Fotoğraf uygun değil' : 'İlk kontrolden geçildi'}</h1>
                <p class="text-slate-500 mb-8">${hasFail 
                    ? 'Lütfen aşağıdaki sorunları düzeltin ve tekrar deneyin.'
                    : 'Ödeme işleminden sonra uzmanımız, %100 uyumlu olduğundan emin olmak için fotoğrafınızı doğrulayacaktır.'
                }</p>
                
                <!-- Issues if any -->
                ${issuesHtml}
                
                <!-- Checklist -->
                <div class="bg-white rounded-2xl border border-slate-200 divide-y divide-slate-100">
                    ${checklistHtml}
                </div>
                
                <!-- Action Buttons -->
                <div class="mt-8 flex flex-col sm:flex-row gap-4">
                    ${!hasFail ? `
                        <button id="proceedToCheckoutBtn" class="flex-1 py-4 px-6 bg-slate-900 hover:bg-slate-800 text-white font-bold text-lg rounded-xl transition-all ${hasWarn ? 'opacity-50 cursor-not-allowed' : ''}" ${hasWarn ? 'disabled' : ''}>
                            Ödeme işlemine geçin
                        </button>
                    ` : ''}
                    <button id="retakePhotoValidationBtn" class="flex-1 py-4 px-6 bg-white hover:bg-slate-50 text-slate-700 font-semibold text-lg rounded-xl border-2 border-slate-200 transition-all">
                        Fotoğrafı tekrar çekin
                    </button>
                </div>
            </div>
            
            <!-- RIGHT COLUMN: Photo Preview -->
            <div class="lg:w-[450px] bg-slate-50 p-8 lg:p-12 flex items-center justify-center border-t lg:border-t-0 lg:border-l border-slate-200">
                <div class="relative">
                    <!-- Dimension Labels -->
                    <div class="absolute -top-8 left-1/2 -translate-x-1/2 text-sm text-slate-500 font-medium">50 mm</div>
                    <div class="absolute top-1/2 -right-12 -translate-y-1/2 text-sm text-slate-500 font-medium transform rotate-90 origin-center">60 mm</div>
                    
                    <!-- Photo Frame -->
                    <div class="relative border-2 border-dashed border-slate-300 p-2 rounded-lg bg-white shadow-lg">
                        <!-- Corner Marks -->
                        <div class="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-slate-400"></div>
                        <div class="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-slate-400"></div>
                        <div class="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-slate-400"></div>
                        <div class="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-slate-400"></div>
                        
                        <img 
                            src="${previewUrl}" 
                            alt="Fotoğraf önizleme"
                            class="w-[200px] h-[240px] object-cover rounded"
                        >
                        
                    </div>
                    
                    <!-- PhotoAiD.com watermark style -->
                    <p class="text-center text-xs text-slate-400 mt-4">BiyometrikFoto.tr</p>
                </div>
            </div>
        </div>
    `;
    
    // Update modal size
    modalContent.className = 'relative bg-white rounded-2xl shadow-2xl max-w-5xl w-full mx-4 max-h-[90vh] overflow-hidden';
    
    // Wire up event handlers
    document.getElementById('closeValidationBtn')?.addEventListener('click', () => {
            closeResultModal();
            window.location.href = '/';
    });
    
    document.getElementById('retakePhotoValidationBtn')?.addEventListener('click', () => {
        closeResultModal();
        window.location.href = '/';
    });
    
    // Handle proceed to checkout button
    const proceedBtn = document.getElementById('proceedToCheckoutBtn');
    if (proceedBtn) {
        proceedBtn.addEventListener('click', () => {
            if (proceedBtn.disabled) return;
            
            // Show processing state
            proceedBtn.disabled = true;
            proceedBtn.innerHTML = `
                <svg class="w-5 h-5 animate-spin inline-block mr-2" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                İşleniyor...
            `;
            
            // Start processing and then show checkout
            processPhotoAndShowCheckout(currentJobId, acknowledgedIssueIds);
        });
    }
    
    // Handle acknowledgement checkboxes
    document.querySelectorAll('.validation-ack-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const issueId = e.target.dataset.issueId;
            if (e.target.checked) {
                if (!acknowledgedIssueIds.includes(issueId)) {
                    acknowledgedIssueIds.push(issueId);
                }
            } else {
                acknowledgedIssueIds = acknowledgedIssueIds.filter(id => id !== issueId);
            }
            
            // Update button state
            const allAcked = document.querySelectorAll('.validation-ack-checkbox').length === 
                            document.querySelectorAll('.validation-ack-checkbox:checked').length;
            if (proceedBtn) {
                proceedBtn.disabled = !allAcked;
                proceedBtn.classList.toggle('opacity-50', !allAcked);
                proceedBtn.classList.toggle('cursor-not-allowed', !allAcked);
            }
        });
    });
    
    resultModal.classList.remove('hidden');
}

// Process photo and then show checkout screen
async function processPhotoAndShowCheckout(jobId, acknowledgedIds) {
    try {
        const response = await fetch(`/process/${jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ acknowledged_issue_ids: acknowledgedIds })
        });
        
        const data = await response.json();
        console.log('[processPhotoAndShowCheckout] Response:', data);
        
        if (data.success && data.job?.processed_url) {
            // Show checkout with processed image
            showProcessedResult(jobId, data.job.processed_url);
        } else if (data.success && data.job) {
            // Fallback - use preview URL
            const previewUrl = data.job.preview_url || `/uploads/${jobId}.jpeg`;
            showProcessedResult(jobId, previewUrl);
        } else {
            alert('Fotoğraf işlenirken bir hata oluştu: ' + (data.error || 'Bilinmeyen hata'));
        }
    } catch (error) {
        console.error('[processPhotoAndShowCheckout] Error:', error);
        alert('Bir hata oluştu. Lütfen tekrar deneyin.');
    }
}

/**
 * Show free download result when payments are disabled (beta mode)
 */
function showFreeDownloadResult(jobId, outputUrl) {
    if (!resultModal) return;
    
    const modalContent = resultModal.querySelector('.relative.bg-white');
    if (!modalContent) return;
    
    const previewUrl = outputUrl.includes('?') ? outputUrl : outputUrl + '?t=' + Date.now();
    
    modalContent.innerHTML = `
        <!-- Header Bar -->
        <div class="bg-emerald-600 text-white px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                </div>
                <span class="font-semibold">BiyometrikFoto.tr</span>
            </div>
            <button id="closeFreeDownloadBtn" class="text-white/70 hover:text-white p-1 rounded transition-colors">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
        
        <!-- Content -->
        <div class="p-8 text-center">
            <!-- Success Icon -->
            <div class="w-20 h-20 mx-auto mb-6 rounded-full bg-emerald-100 flex items-center justify-center">
                <svg class="w-10 h-10 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
            </div>
            
            <!-- Title -->
            <h1 class="text-2xl font-bold text-slate-900 mb-2">Fotoğrafınız Hazır! 🎉</h1>
            <p class="text-slate-500 mb-6">Türkiye biyometrik standartlarına uygun fotoğrafınız oluşturuldu.</p>
            
            <!-- Beta Notice -->
            <div class="inline-flex items-center gap-2 px-4 py-2 bg-amber-50 border border-amber-200 rounded-full mb-8">
                <svg class="w-4 h-4 text-amber-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                </svg>
                <span class="text-amber-800 font-medium text-sm">Beta döneminde indirme ücretsiz!</span>
            </div>
            
            <!-- Preview -->
            <div class="max-w-xs mx-auto mb-8">
                <div class="relative rounded-2xl overflow-hidden shadow-xl border-4 border-white">
                    <img src="${previewUrl}" alt="Biyometrik Fotoğraf" class="w-full aspect-[5/6] object-cover">
                    <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-4">
                        <div class="text-white text-sm font-medium">50 × 60 mm</div>
                    </div>
                </div>
            </div>
            
            <!-- Download Button -->
            <a 
                href="/api/download/${jobId}" 
                download="biyometrik_foto.png"
                class="inline-flex items-center justify-center gap-3 px-8 py-4 bg-emerald-600 text-white rounded-xl font-bold text-lg hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-600/30 hover:shadow-xl hover:shadow-emerald-600/40 hover:-translate-y-0.5 active:translate-y-0"
            >
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                </svg>
                İndir
            </a>
            
            <!-- Upload New -->
            <div class="mt-6">
                <button id="uploadNewBetaBtn" class="text-slate-500 hover:text-emerald-600 transition-colors text-sm">
                    ← Yeni fotoğraf yükle
                </button>
            </div>
        </div>
    `;
    
    // Update modal size
    modalContent.className = 'relative bg-white rounded-2xl shadow-2xl max-w-lg w-full mx-4 overflow-hidden';
    
    // Event handlers
    const closeBtn = document.getElementById('closeFreeDownloadBtn');
    const uploadNewBtn = document.getElementById('uploadNewBetaBtn');
    
    if (closeBtn) {
        closeBtn.onclick = () => {
            resultModal.classList.add('hidden');
        };
    }
    
    if (uploadNewBtn) {
        uploadNewBtn.onclick = () => {
            resultModal.classList.add('hidden');
            resetUploadState();
        };
    }
    
    // Show modal
    resultModal.classList.remove('hidden');
}

function showProcessedResult(jobId, outputUrl) {
    if (!resultModal) return;
    
    // Mark photo as processed and store URL for download
    isPhotoProcessed = true;
    processedPhotoUrl = outputUrl;
    
    // Debug logging
    console.log('[DEBUG] showProcessedResult:');
    console.log('  - jobId:', jobId);
    console.log('  - outputUrl:', outputUrl);
    console.log('  - paymentsEnabled:', isPaymentsEnabled());
    
    // If payments disabled, show free download UI
    if (!isPaymentsEnabled()) {
        showFreeDownloadResult(jobId, outputUrl);
        return;
    }
    
    // Get prices from Stripe config
    const prices = stripeConfig?.prices || {
        digital: { display: '₺100', amount_tl: 100, name: 'Dijital', description: 'Anında indir' },
        digital_print: { display: '₺200', amount_tl: 200, name: 'Dijital + Baskı', description: 'Baskı + kargo' }
    };
    
    // Build premium checkout page
    const modalContent = resultModal.querySelector('.relative.bg-white');
    if (!modalContent) return;
    
    // Add cache busting to image URL
    const previewUrl = outputUrl.includes('?') ? outputUrl : outputUrl + '?t=' + Date.now();
    
    modalContent.innerHTML = `
        <!-- Checkout Header Bar -->
        <div class="bg-slate-900 text-white px-6 py-3 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
                    </svg>
                </div>
                <span class="font-semibold text-sm">BiyometrikFoto.tr</span>
            </div>
            <div class="flex items-center gap-2 text-xs text-slate-300">
                <svg class="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                </svg>
                <span>SSL Güvenli Ödeme</span>
            </div>
            <button id="closeCheckoutBtn" class="text-slate-400 hover:text-white p-1 rounded transition-colors">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
        
        <!-- Main Content: 2-Column Layout -->
        <div class="flex flex-col lg:flex-row" style="min-height: calc(100vh - 200px); max-height: calc(90vh - 52px);">
            
            <!-- LEFT COLUMN: Options & Info -->
            <div class="flex-1 p-6 lg:p-10 overflow-y-auto bg-white">
                
                <!-- Success Header -->
                <div class="mb-8">
                    <div class="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-full mb-4">
                        <div class="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
                            <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                            </svg>
                        </div>
                        <span class="text-emerald-700 font-medium text-sm">Türkiye biyometrik standartlarına uygun</span>
                    </div>
                    <h1 class="text-2xl lg:text-3xl font-bold text-slate-900 mb-2">Fotoğraf Onaylandı</h1>
                    <p class="text-slate-500">İndirmek için ödeme gerekli. Ödeme sonrası fotoğraf netleşir ve anında indirilebilir.</p>
                </div>
                
                <!-- Section 1: Package Selection -->
                <div class="mb-8">
                    <div class="flex items-center gap-3 mb-5">
                        <span class="w-7 h-7 rounded-full bg-emerald-600 text-white text-sm flex items-center justify-center font-bold">1</span>
                        <h2 class="text-lg font-bold text-slate-900">Paket seçin</h2>
                    </div>
                    
                    <!-- Digital Option -->
                    <label class="block mb-4 cursor-pointer group">
                        <div id="optionDigital" class="relative p-5 border-2 border-emerald-500 bg-emerald-50/50 rounded-2xl transition-all duration-200 hover:shadow-lg">
                            <!-- Popular Badge -->
                            <div class="absolute -top-3 left-5 px-3 py-1 bg-amber-400 text-amber-900 text-xs font-bold rounded-full shadow-sm">
                                En popüler
                            </div>
                            <div class="flex items-start gap-4">
                                <div class="mt-1">
                                    <input type="radio" name="plan" value="digital" checked class="w-5 h-5 text-emerald-600 border-2 border-slate-300 focus:ring-emerald-500 focus:ring-offset-0">
                                </div>
                                <div class="flex-1">
                                    <div class="flex justify-between items-start mb-2">
                                        <h3 class="text-lg font-bold text-slate-900">Dijital</h3>
                                        <div class="text-right">
                                            <span class="text-2xl font-bold text-emerald-600">${prices.digital.display}</span>
                                        </div>
                                    </div>
                                    <p class="text-sm text-slate-600 mb-3">Dijital dosya + anında indirme.</p>
                                    <ul class="space-y-2">
                                        <li class="flex items-center gap-2 text-sm text-slate-600">
                                            <svg class="w-4 h-4 text-emerald-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                            </svg>
                                            Anında indirme
                                        </li>
                                        <li class="flex items-center gap-2 text-sm text-slate-600">
                                            <svg class="w-4 h-4 text-emerald-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                            </svg>
                                            Opsiyonel e-posta ile link
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </label>
                    
                    <!-- Digital + Print Option -->
                    <label class="block cursor-pointer group">
                        <div id="optionPrint" class="relative p-5 border-2 border-slate-200 bg-white rounded-2xl transition-all duration-200 hover:border-slate-300 hover:shadow-lg">
                            <div class="flex items-start gap-4">
                                <div class="mt-1">
                                    <input type="radio" name="plan" value="digital_print" class="w-5 h-5 text-emerald-600 border-2 border-slate-300 focus:ring-emerald-500 focus:ring-offset-0">
                                </div>
                                <div class="flex-1">
                                    <div class="flex justify-between items-start mb-2">
                                        <h3 class="text-lg font-bold text-slate-900">Dijital + Baskı</h3>
                                        <div class="text-right">
                                            <span class="text-2xl font-bold text-slate-900">${prices.digital_print.display}</span>
                                        </div>
                                    </div>
                                    <p class="text-sm text-slate-600 mb-3">4 adet baskı + Türkiye içi kargo.</p>
                                    <ul class="space-y-2">
                                        <li class="flex items-center gap-2 text-sm text-slate-600">
                                            <svg class="w-4 h-4 text-emerald-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                            </svg>
                                            Dijitalin tüm özellikleri
                                        </li>
                                        <li class="flex items-center gap-2 text-sm text-slate-600">
                                            <svg class="w-4 h-4 text-emerald-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                            </svg>
                                            4 adet profesyonel baskı
                                        </li>
                                        <li class="flex items-center gap-2 text-sm text-slate-600">
                                            <svg class="w-4 h-4 text-emerald-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                            </svg>
                                            Türkiye içi ücretsiz kargo
                                        </li>
                                    </ul>
                                    <p class="mt-3 text-xs text-slate-400 flex items-center gap-1">
                                        <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path>
                                        </svg>
                                        Kargo adresi ödeme sonrası alınır.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </label>
                </div>
                
                <!-- Section 2: Customer Info -->
                <div class="mb-8">
                    <div class="flex items-center gap-3 mb-5">
                        <span class="w-7 h-7 rounded-full bg-slate-200 text-slate-600 text-sm flex items-center justify-center font-bold">2</span>
                        <h2 class="text-lg font-bold text-slate-900" id="infoSectionTitle">Bilgileriniz</h2>
                        <span class="text-sm text-slate-400" id="infoSectionBadge">(opsiyonel)</span>
                    </div>
                    
                    <!-- Email only - for Digital -->
                    <div id="emailOnlySection" class="bg-slate-50 rounded-xl p-5">
                        <label class="block text-sm font-medium text-slate-700 mb-2">E-posta adresi</label>
                        <input 
                            type="email" 
                            id="checkoutEmail" 
                            placeholder="ornek@mail.com"
                            class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                        >
                        <p class="text-xs text-slate-400 mt-2">İndirme linkini e-posta ile de gönderebiliriz.</p>
                    </div>
                    
                    <!-- Full address form - for Digital + Print -->
                    <div id="addressFormSection" class="hidden bg-slate-50 rounded-xl p-5 space-y-4">
                        <p class="text-sm text-slate-600 mb-4 flex items-center gap-2">
                            <svg class="w-4 h-4 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                            </svg>
                            Baskı için teslimat bilgileri gereklidir
                        </p>
                        
                        <!-- Name fields -->
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-slate-700 mb-2">Ad <span class="text-red-500">*</span></label>
                                <input 
                                    type="text" 
                                    id="shippingFirstName" 
                                    placeholder="Adınız"
                                    required
                                    class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                                >
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-slate-700 mb-2">Soyad <span class="text-red-500">*</span></label>
                                <input 
                                    type="text" 
                                    id="shippingLastName" 
                                    placeholder="Soyadınız"
                                    required
                                    class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                                >
                            </div>
                        </div>
                        
                        <!-- Address -->
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Adres <span class="text-red-500">*</span></label>
                            <input 
                                type="text" 
                                id="shippingAddress" 
                                placeholder="Mahalle, sokak, bina no, daire no"
                                required
                                class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                            >
                        </div>
                        
                        <!-- City, Postal -->
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-slate-700 mb-2">Şehir <span class="text-red-500">*</span></label>
                                <input 
                                    type="text" 
                                    id="shippingCity" 
                                    placeholder="İstanbul"
                                    required
                                    class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                                >
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-slate-700 mb-2">Posta Kodu <span class="text-red-500">*</span></label>
                                <input 
                                    type="text" 
                                    id="shippingPostalCode" 
                                    placeholder="34000"
                                    required
                                    class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                                >
                            </div>
                        </div>
                        
                        <!-- Phone -->
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Telefon <span class="text-red-500">*</span></label>
                            <input 
                                type="tel" 
                                id="shippingPhone" 
                                placeholder="05XX XXX XX XX"
                                required
                                class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                            >
                        </div>
                        
                        <!-- Email -->
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">E-posta <span class="text-red-500">*</span></label>
                            <input 
                                type="email" 
                                id="shippingEmail" 
                                placeholder="ornek@mail.com"
                                required
                                class="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                            >
                            <p class="text-xs text-slate-400 mt-2">Sipariş onayı ve kargo takip bilgisi bu adrese gönderilecek.</p>
                        </div>
                    </div>
                </div>
                
                <!-- Trust Strip -->
                <div class="flex flex-wrap items-center gap-4 py-4 border-t border-slate-100">
                    <div class="flex items-center gap-2 text-sm text-slate-500">
                        <svg class="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                        </svg>
                        <span>SSL güvenli</span>
                    </div>
                    <div class="flex items-center gap-2 text-sm text-slate-500">
                        <svg class="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                        </svg>
                        <span>Kabul garantisi</span>
                    </div>
                    <div class="flex items-center gap-2 text-sm text-slate-500">
                        <svg class="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"></path>
                        </svg>
                        <span>Saniyeler içinde hazır</span>
                    </div>
                </div>
                
                <!-- Back Link -->
                <button id="backToUploadBtn" class="mt-4 text-slate-400 hover:text-slate-600 text-sm flex items-center gap-1 transition-colors">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                    </svg>
                    Farklı fotoğraf yükle
                </button>
            </div>
            
            <!-- RIGHT COLUMN: Summary & CTA -->
            <div class="lg:w-[420px] bg-slate-50 p-6 lg:p-8 border-t lg:border-t-0 lg:border-l border-slate-200 lg:sticky lg:top-0">
                
                <!-- Summary Card -->
                <div class="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 mb-6">
                    <!-- Header -->
                    <div class="flex items-center justify-between mb-4 pb-4 border-b border-slate-100">
                        <div>
                            <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Özet</h3>
                            <p class="text-xs text-slate-300">Sipariş: ${jobId.substring(0, 8)}</p>
                        </div>
                        <div class="w-10 h-10 bg-emerald-100 rounded-full flex items-center justify-center">
                            <svg class="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                        </div>
                    </div>
                    
                    <!-- Product Info -->
                    <div class="mb-4">
                        <h4 class="font-bold text-slate-900">Türkiye Biyometrik Fotoğraf</h4>
                        <p class="text-sm text-slate-500">Boyut: 50 × 60 mm</p>
                    </div>
                    
                    <!-- Preview Card -->
                    <div class="relative bg-slate-100 rounded-xl p-3 mb-5">
                        <div class="aspect-[5/6] bg-slate-200 rounded-lg overflow-hidden relative">
                            <img 
                                src="${previewUrl}" 
                                alt="Fotoğraf önizleme"
                                class="w-full h-full object-cover"
                                style="filter: blur(10px); transform: scale(1.05);"
                            >
                            <!-- Lock Overlay -->
                            <div class="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/30 backdrop-blur-[2px]">
                                <div class="w-14 h-14 bg-white rounded-2xl shadow-lg flex items-center justify-center mb-3">
                                    <svg class="w-7 h-7 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                                    </svg>
                                </div>
                                <span class="text-white text-sm font-medium drop-shadow">Kilitli önizleme</span>
                            </div>
                        </div>
                        <p class="text-xs text-center text-slate-400 mt-2">Ödeme sonrası netleşir</p>
                    </div>
                    
                    <!-- Line Items -->
                    <div class="space-y-3 mb-4 pb-4 border-b border-slate-100">
                        <div class="flex justify-between text-sm">
                            <span class="text-slate-500" id="summaryProductName">Dijital versiyon</span>
                            <span class="text-slate-700 font-medium" id="summaryProductPrice">${prices.digital.display}</span>
                        </div>
                        <div id="summaryShippingLine" class="hidden flex justify-between text-sm">
                            <span class="text-slate-500">Kargo</span>
                            <span class="text-emerald-600 font-medium">Ücretsiz</span>
                        </div>
                    </div>
                    
                    <!-- Total -->
                    <div class="flex justify-between items-center mb-6">
                        <span class="text-slate-700 font-semibold">Toplam</span>
                        <span class="text-3xl font-bold text-emerald-600" id="summaryTotal">${prices.digital.display}</span>
                    </div>
                    
                    <!-- CTA Button -->
                    <button 
                        id="payNowBtn"
                        class="w-full py-4 bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-lg rounded-xl transition-all duration-200 shadow-lg shadow-emerald-600/20 hover:shadow-emerald-600/30 hover:scale-[1.02] flex items-center justify-center gap-2"
                    >
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                        </svg>
                        Şimdi ödeyin
                    </button>
                </div>
                
                <!-- Trust Badges Grid -->
                <div class="grid grid-cols-2 gap-3 mb-4">
                    <div class="bg-white rounded-xl p-3 border border-slate-100 flex items-center gap-2">
                        <svg class="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                        </svg>
                        <span class="text-xs text-slate-600 font-medium">SSL Secure</span>
                    </div>
                    <div class="bg-white rounded-xl p-3 border border-slate-100 flex items-center gap-2">
                        <svg class="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M3 6a3 3 0 013-3h10a1 1 0 01.8 1.6L14.25 8l2.55 3.4A1 1 0 0116 13H6a1 1 0 00-1 1v3a1 1 0 11-2 0V6z" clip-rule="evenodd"></path>
                        </svg>
                        <span class="text-xs text-slate-600 font-medium">TR Standardı</span>
                    </div>
                    <div class="bg-white rounded-xl p-3 border border-slate-100 flex items-center gap-2">
                        <svg class="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"></path>
                        </svg>
                        <span class="text-xs text-slate-600 font-medium">ICAO Uyumlu*</span>
                    </div>
                    <div class="bg-white rounded-xl p-3 border border-slate-100 flex items-center gap-2">
                        <svg class="w-5 h-5" viewBox="0 0 32 32" fill="none">
                            <rect width="32" height="32" rx="6" fill="#635BFF"/>
                            <path d="M15.5 10.5h-3v8h2v-3h1c1.7 0 3-1.3 3-2.5s-1.3-2.5-3-2.5zm0 3.5h-1v-2h1c.6 0 1 .4 1 1s-.4 1-1 1zm6.5-3.5h-2v8h2v-8z" fill="white"/>
                        </svg>
                        <span class="text-xs text-slate-600 font-medium">Stripe ile ödeme</span>
                    </div>
                </div>
                
                <!-- Footnote -->
                <p class="text-[10px] text-slate-400 text-center leading-relaxed">
                    *ICAO genel biyometrik prensipler ile uyumlu; nihai kabul kurum takdirindedir.
                    Ödeme sonrası: indir + istersen e-posta ile link.
                </p>
            </div>
        </div>
        
        <!-- Mobile Sticky CTA -->
        <div class="lg:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4 shadow-2xl z-50" id="mobileStickyCta">
            <div class="flex items-center justify-between mb-3">
                <div>
                    <span class="text-xs text-slate-400">Toplam</span>
                    <span class="block text-xl font-bold text-emerald-600" id="mobileTotalPrice">${prices.digital.display}</span>
                </div>
                <div class="flex items-center gap-2 text-xs text-slate-400">
                    <svg class="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                    </svg>
                    <span>Güvenli ödeme</span>
                </div>
            </div>
            <button 
                id="mobilePayBtn"
                class="w-full py-4 bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-lg rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg"
            >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                </svg>
                Şimdi ödeyin
            </button>
        </div>
    `;
    
    // Update modal size for checkout layout
    modalContent.className = 'relative bg-white rounded-2xl shadow-2xl max-w-6xl w-full mx-4 max-h-[90vh] overflow-hidden';
    
    // Wire up event handlers
    setupCheckoutEventHandlers(jobId, prices);
    
    resultModal.classList.remove('hidden');
}

function setupCheckoutEventHandlers(jobId, prices) {
    // Plan selection
    const optionDigital = document.getElementById('optionDigital');
    const optionPrint = document.getElementById('optionPrint');
    const radioDigital = document.querySelector('input[value="digital"]');
    const radioPrint = document.querySelector('input[value="digital_print"]');
    
    function updateSelection(plan) {
        selectedPlan = plan;
        
        // Update card styles - premium version
        if (plan === 'digital') {
            optionDigital.className = 'relative p-5 border-2 border-emerald-500 bg-emerald-50/50 rounded-2xl transition-all duration-200 hover:shadow-lg';
            optionPrint.className = 'relative p-5 border-2 border-slate-200 bg-white rounded-2xl transition-all duration-200 hover:border-slate-300 hover:shadow-lg';
            radioDigital.checked = true;
        } else {
            optionDigital.className = 'relative p-5 border-2 border-slate-200 bg-white rounded-2xl transition-all duration-200 hover:border-slate-300 hover:shadow-lg';
            optionPrint.className = 'relative p-5 border-2 border-emerald-500 bg-emerald-50/50 rounded-2xl transition-all duration-200 hover:shadow-lg';
            radioPrint.checked = true;
        }
        
        // Toggle form sections
        const emailOnlySection = document.getElementById('emailOnlySection');
        const addressFormSection = document.getElementById('addressFormSection');
        const infoSectionTitle = document.getElementById('infoSectionTitle');
        const infoSectionBadge = document.getElementById('infoSectionBadge');
        
        if (plan === 'digital') {
            // Show email only for digital
            emailOnlySection?.classList.remove('hidden');
            addressFormSection?.classList.add('hidden');
            if (infoSectionTitle) infoSectionTitle.textContent = 'Bilgileriniz';
            if (infoSectionBadge) {
                infoSectionBadge.textContent = '(opsiyonel)';
                infoSectionBadge.className = 'text-sm text-slate-400';
            }
        } else {
            // Show full address form for digital + print
            emailOnlySection?.classList.add('hidden');
            addressFormSection?.classList.remove('hidden');
            if (infoSectionTitle) infoSectionTitle.textContent = 'Teslimat Bilgileri';
            if (infoSectionBadge) {
                infoSectionBadge.textContent = '(zorunlu)';
                infoSectionBadge.className = 'text-sm text-red-500 font-medium';
            }
        }
        
        // Update summary
        const summaryProductName = document.getElementById('summaryProductName');
        const summaryProductPrice = document.getElementById('summaryProductPrice');
        const summaryShippingLine = document.getElementById('summaryShippingLine');
        const summaryTotal = document.getElementById('summaryTotal');
        const mobileTotalPrice = document.getElementById('mobileTotalPrice');
        
        if (plan === 'digital') {
            if (summaryProductName) summaryProductName.textContent = 'Dijital versiyon';
            if (summaryProductPrice) summaryProductPrice.textContent = prices.digital.display;
            summaryShippingLine?.classList.add('hidden');
            if (summaryTotal) summaryTotal.textContent = prices.digital.display;
            if (mobileTotalPrice) mobileTotalPrice.textContent = prices.digital.display;
        } else {
            if (summaryProductName) summaryProductName.textContent = 'Dijital + Baskı paketi';
            if (summaryProductPrice) summaryProductPrice.textContent = prices.digital_print.display;
            summaryShippingLine?.classList.remove('hidden');
            if (summaryTotal) summaryTotal.textContent = prices.digital_print.display;
            if (mobileTotalPrice) mobileTotalPrice.textContent = prices.digital_print.display;
        }
    }
    
    // Click handlers for option cards
    optionDigital?.addEventListener('click', () => updateSelection('digital'));
    optionPrint?.addEventListener('click', () => updateSelection('digital_print'));
    
    // Radio change handlers
    radioDigital?.addEventListener('change', () => updateSelection('digital'));
    radioPrint?.addEventListener('change', () => updateSelection('digital_print'));
    
    // Email inputs - both for digital and digital+print
    const emailInput = document.getElementById('checkoutEmail');
    const shippingEmailInput = document.getElementById('shippingEmail');
    
    emailInput?.addEventListener('input', (e) => {
        customerEmail = e.target.value;
    });
    shippingEmailInput?.addEventListener('input', (e) => {
        customerEmail = e.target.value;
    });
    
    // Pay button handlers
    const payNowBtn = document.getElementById('payNowBtn');
    const mobilePayBtn = document.getElementById('mobilePayBtn');
    
    // Validate shipping form for digital+print
    const validateShippingForm = () => {
        const requiredFields = [
            { id: 'shippingFirstName', name: 'Ad' },
            { id: 'shippingLastName', name: 'Soyad' },
            { id: 'shippingAddress', name: 'Adres' },
            { id: 'shippingCity', name: 'Şehir' },
            { id: 'shippingPostalCode', name: 'Posta Kodu' },
            { id: 'shippingPhone', name: 'Telefon' },
            { id: 'shippingEmail', name: 'E-posta' }
        ];
        
        const missingFields = [];
        const shippingData = {};
        
        for (const field of requiredFields) {
            const input = document.getElementById(field.id);
            const value = input?.value?.trim() || '';
            
            if (!value) {
                missingFields.push(field.name);
                input?.classList.add('border-red-500', 'ring-2', 'ring-red-200');
            } else {
                input?.classList.remove('border-red-500', 'ring-2', 'ring-red-200');
                shippingData[field.id.replace('shipping', '').toLowerCase()] = value;
            }
        }
        
        if (missingFields.length > 0) {
            return { valid: false, message: `Lütfen zorunlu alanları doldurun: ${missingFields.join(', ')}`, data: null };
        }
        
        // Validate email format
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(shippingData.email)) {
            document.getElementById('shippingEmail')?.classList.add('border-red-500', 'ring-2', 'ring-red-200');
            return { valid: false, message: 'Lütfen geçerli bir e-posta adresi girin', data: null };
        }
        
        return { valid: true, message: '', data: shippingData };
    };
    
    const handlePay = () => {
        // Validate shipping form for digital+print
        if (selectedPlan === 'digital_print') {
            const validation = validateShippingForm();
            if (!validation.valid) {
                alert(validation.message);
                return;
            }
            // Store shipping data for later use (will be passed to checkout)
            window.shippingData = validation.data;
            customerEmail = validation.data.email;
            console.log('[CHECKOUT] Shipping data validated:', validation.data);
        }
        
        console.log('[CHECKOUT] Initiating payment:', { plan: selectedPlan, email: customerEmail, jobId });
        
        // Show loading state on buttons
        if (payNowBtn) {
            payNowBtn.disabled = true;
            payNowBtn.innerHTML = `
                <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>İşleniyor...</span>
            `;
        }
        if (mobilePayBtn) {
            mobilePayBtn.disabled = true;
            mobilePayBtn.innerHTML = `
                <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>İşleniyor...</span>
            `;
        }
        
        initiateCheckout(jobId, selectedPlan, window.shippingData);
    };
    
    payNowBtn?.addEventListener('click', handlePay);
    mobilePayBtn?.addEventListener('click', handlePay);
    
    // Close button
    document.getElementById('closeCheckoutBtn')?.addEventListener('click', () => {
        closeResultModal();
    });
    
    // Back to upload button
    document.getElementById('backToUploadBtn')?.addEventListener('click', () => {
        closeResultModal();
        window.location.href = '/';
    });
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
const closeResultModalBtn = document.getElementById('closeResultModalBtn');

if (resultOverlay) {
    resultOverlay.addEventListener('click', closeResultModal);
}
if (closeResultModalBtn) {
    closeResultModalBtn.addEventListener('click', () => {
        closeResultModal();
        window.location.href = '/';
    });
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
// Note: continueBtn click handler is set dynamically in showResultScreen() and showProcessedResult()
// Do not add a global event listener here as it would conflict with those handlers

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

// ============================================================================
// Page Navigation (Landing → Upload → Processing → Result)
// ============================================================================
const landingPage = document.getElementById('landingPage');
const uploadPage = document.getElementById('uploadPage');
const productModal = document.getElementById('productModal');
const startUploadBtn = document.getElementById('startUploadBtn');
const backToLandingBtn = document.getElementById('backToLandingBtn');
const dropZone = document.getElementById('dropZone');
const selectedFilePreview = document.getElementById('selectedFilePreview');
const selectedFileImage = document.getElementById('selectedFileImage');
const selectedFileName = document.getElementById('selectedFileName');
const selectedFileSize = document.getElementById('selectedFileSize');
const removeFileBtn = document.getElementById('removeFileBtn');
const uploadBtn = document.getElementById('uploadBtn');

function showLandingPage() {
    if (landingPage) landingPage.classList.remove('hidden');
    if (uploadPage) uploadPage.classList.add('hidden');
    // Reset state
    resetUploadState();
}

function showUploadPage() {
    if (landingPage) landingPage.classList.add('hidden');
    if (uploadPage) uploadPage.classList.remove('hidden');
}

function resetUploadState() {
    if (photoInput) photoInput.value = '';
    if (selectedFilePreview) selectedFilePreview.classList.add('hidden');
    if (uploadBtn) uploadBtn.classList.add('hidden');
    if (dropZone) dropZone.classList.remove('hidden');
    if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
        currentImageUrl = null;
    }
    isPhotoProcessed = false;
    processedPhotoUrl = null;
    acknowledgedIssueIds = [];
    backendDonePayload = null;
    isPaid = false;
}

// Start Upload Buttons (Landing → Upload)
if (startUploadBtn) {
    startUploadBtn.addEventListener('click', showUploadPage);
}

// Second CTA button at bottom of page
const startUploadBtn2 = document.getElementById('startUploadBtn2');
if (startUploadBtn2) {
    startUploadBtn2.addEventListener('click', showUploadPage);
}

// Back Button (Upload → Landing)
if (backToLandingBtn) {
    backToLandingBtn.addEventListener('click', showLandingPage);
}

// ============================================================================
// Drop Zone & File Selection
// ============================================================================
if (dropZone && photoInput) {
    // Click to select file
    dropZone.addEventListener('click', () => {
        photoInput.click();
    });
    
    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-emerald-500', 'bg-emerald-50');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-emerald-500', 'bg-emerald-50');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-emerald-500', 'bg-emerald-50');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            photoInput.files = files;
            handleFileSelected(files[0]);
        }
    });
}

// File input change
if (photoInput) {
    photoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelected(file);
        }
    });
}

function handleFileSelected(file) {
    // Show preview
    if (selectedFilePreview && selectedFileImage && selectedFileName && selectedFileSize) {
        currentImageUrl = URL.createObjectURL(file);
        selectedFileImage.src = currentImageUrl;
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = formatFileSize(file.size);
        
        dropZone.classList.add('hidden');
        selectedFilePreview.classList.remove('hidden');
        uploadBtn.classList.remove('hidden');
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Remove file button
if (removeFileBtn) {
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadState();
        dropZone.classList.remove('hidden');
    });
}

// ============================================================================
// Product Selection Modal
// ============================================================================
const productRadios = document.querySelectorAll('input[name="productType"]');
const proceedToPaymentBtn = document.getElementById('proceedToPaymentBtn');

productRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        if (proceedToPaymentBtn) {
            proceedToPaymentBtn.disabled = false;
        }
    });
});

function showProductModal() {
    if (productModal) {
        productModal.classList.remove('hidden');
    }
}

function closeProductModal() {
    if (productModal) {
        productModal.classList.add('hidden');
        // Reset selection
        productRadios.forEach(radio => radio.checked = false);
        if (proceedToPaymentBtn) proceedToPaymentBtn.disabled = true;
    }
}

if (proceedToPaymentBtn) {
    proceedToPaymentBtn.addEventListener('click', () => {
        const selectedProduct = document.querySelector('input[name="productType"]:checked');
        if (selectedProduct) {
            // For now, just show an alert (payment not implemented)
            alert('Ödeme sayfası yakında eklenecek!\n\nSeçilen: ' + 
                (selectedProduct.value === 'digital' ? 'Dijital fotoğraf' : 'Baskı + kargo'));
        }
    });
}

// ============================================================================
// Override close result modal to go back to landing
// ============================================================================
const originalCloseResultModal = closeResultModal;
closeResultModal = function() {
    if (resultModal) {
        resultModal.classList.add('hidden');
    }
    showLandingPage();
};

// ============================================================================
// Initialize on page load
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Load Stripe configuration
    loadStripeConfig();
    
    // Check if returning from payment
    const urlParams = new URLSearchParams(window.location.search);
    const jobIdFromUrl = urlParams.get('job_id');
    
    if (jobIdFromUrl) {
        // Restore state for this job
        currentJobId = jobIdFromUrl;
        console.log('[INIT] Job ID from URL:', jobIdFromUrl);
    }
});

