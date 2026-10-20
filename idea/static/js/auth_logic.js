// auth_logic.js
// Handles Firebase Authentication and API request signing
// DEPENDS ON: auth.js (which defines FIREBASE_CONFIG)

console.log("Loading Auth Logic...");

const devAuthConfig = window.IDEA_DEV_AUTH || { enabled: false };
const isDevAuthEnabled = Boolean(devAuthConfig.enabled);

// 1. Config Loading Logic
if (!isDevAuthEnabled && typeof FIREBASE_CONFIG === 'undefined' && typeof firebaseConfigObj === 'undefined') {
    console.warn("FIREBASE_CONFIG is missing. Auth may fail.");
} else if (typeof FIREBASE_CONFIG === 'string') {
    // If it's a string (from the deploy script), parse it.
    try {
        window.firebaseConfigObj = JSON.parse(FIREBASE_CONFIG);
    } catch (e) {
        console.error("Error parsing FIREBASE_CONFIG:", e);
    }
} else if (typeof FIREBASE_CONFIG === 'object') {
    window.firebaseConfigObj = FIREBASE_CONFIG;
}

// 2. Initialize Firebase
if (!isDevAuthEnabled && window.firebaseConfigObj && typeof firebase !== 'undefined' && (!firebase.apps || !firebase.apps.length)) {
    console.log("Initializing Firebase with config:", window.firebaseConfigObj.projectId);
    firebase.initializeApp(window.firebaseConfigObj);
} else if (!isDevAuthEnabled && !window.firebaseConfigObj) {
    console.error("Firebase config not found! Authentication will fail.");
}

// Global user state
window.currentUser = null;
let idToken = null;

// DOM Elements
const authContainer = document.getElementById('authContainer');
let loginOverlay = document.getElementById('loginOverlay');

function getSafeUserImageUrl(value, origin = window.location.origin) {
    const fallback = null;
    const rawValue = String(value || '').trim();
    if (!rawValue || rawValue.startsWith('//')) {
        return fallback;
    }
    try {
        const baseUrl = new URL(origin);
        const parsedUrl = new URL(rawValue, baseUrl);
        const isSameOriginHttp = parsedUrl.protocol === 'http:' && parsedUrl.origin === baseUrl.origin;
        if (parsedUrl.protocol === 'https:' || isSameOriginHttp) {
            return parsedUrl.href;
        }
    } catch (_) {
        // Invalid or active URLs use initials instead of an image.
    }
    return fallback;
}

function isInternalRequestUrl(value, origin = window.location.origin) {
    try {
        const requestValue = typeof Request !== 'undefined' && value instanceof Request
            ? value.url
            : value;
        const baseUrl = new URL(origin);
        return new URL(String(requestValue), baseUrl).origin === baseUrl.origin;
    } catch (_) {
        return false;
    }
}

// 3. Create Login Modal if missing
if (!loginOverlay) {
    loginOverlay = document.createElement('div');
    loginOverlay.id = 'loginOverlay';
    loginOverlay.className = 'auth-overlay';
    loginOverlay.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.98); z-index: 10000;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        backdrop-filter: blur(10px);
    `;
    loginOverlay.innerHTML = `
        <div class="auth-panel text-center p-5" style="max-width: 440px;">
            <h2 class="mb-3 fw-bold">Idea Evolution</h2>
            <p class="text-muted mb-4">Sign in to access your evolution workspace.</p>
            <button id="googleSignInBtn" class="btn btn-primary btn-lg w-100 shadow-sm">
                <i class="fab fa-google me-2"></i> Sign in with Google
            </button>
            <p class="mt-3 text-muted small mb-0">Your experiments and provider keys stay in your workspace.</p>
            <p id="loginError" class="text-danger small mt-3 mb-0" role="alert" hidden></p>
        </div>
    `;
    document.body.appendChild(loginOverlay);

    // Add event listener
    setTimeout(() => {
        const btn = document.getElementById('googleSignInBtn');
        if (btn) btn.addEventListener('click', signInWithGoogle);
    }, 100);
}

// 4. Auth Functions
async function signInWithGoogle() {
    if (isDevAuthEnabled) {
        await activateDevAuth();
        return;
    }

    const button = document.getElementById('googleSignInBtn');
    let feedback = document.getElementById('loginError');
    if (!feedback) {
        feedback = document.createElement('p');
        feedback.id = 'loginError';
        feedback.className = 'text-danger small mt-3';
        feedback.setAttribute('role', 'alert');
        loginOverlay.appendChild(feedback);
    }
    feedback.hidden = true;
    if (button) button.disabled = true;
    try {
        if (typeof firebase === 'undefined' || !firebase.apps?.length) {
            throw new Error('Sign-in is unavailable. Refresh the page and try again.');
        }
        const provider = new firebase.auth.GoogleAuthProvider();
        await firebase.auth().signInWithPopup(provider);
    } catch (error) {
        if (error.code === 'auth/popup-closed-by-user' || error.code === 'auth/cancelled-popup-request') return;
        feedback.textContent = error.code === 'auth/popup-blocked'
            ? 'Your browser blocked the sign-in window. Allow pop-ups for this site, then try again.'
            : error.code === 'auth/network-request-failed'
                ? 'Sign-in could not connect. Check your connection and try again.'
                : 'Sign-in is unavailable. Refresh the page and try again.';
        feedback.hidden = false;
    } finally {
        if (button) button.disabled = false;
    }
}

async function signOut() {
    try {
        if (isDevAuthEnabled) {
            window.location.reload();
            return;
        }
        await firebase.auth().signOut();
        // Force reload to clear state
        window.location.reload();
    } catch (error) {
        console.error("Logout failed:", error);
    }
}

// 5. Auth Observer
async function completeSignIn(user) {
    if (window.__workspaceUserId && window.__workspaceUserId !== user.uid) {
        window.location.reload();
        return;
    }
    window.__workspaceUserId = user.uid;
    console.log("User signed in:", user.email);
    window.currentUser = user;
    idToken = null;
    try {
        idToken = await user.getIdToken();
    } catch (e) {
        console.error("Failed to get ID token", e);
        completeSignOut();
        const error = document.getElementById('loginError');
        if (error) { error.textContent = 'Your session could not connect. Please sign in again.'; error.hidden = false; }
        return;
    }
    document.querySelector('.layout-container')?.removeAttribute('inert');

    // Hide overlay
    loginOverlay.style.display = 'none';
    document.body.style.overflow = 'auto';

    // Check API Key Status
    checkApiKeyStatus();

    // Check if there's a running evolution and show banner
    if (typeof checkRunningEvolution === 'function') {
        checkRunningEvolution();
    }

    // Render Header UI
    renderUserUI(user);

    // Refresh template lists now that we have auth
    if (typeof loadExistingTemplates === 'function') {
        loadExistingTemplates();
    }
    if (typeof loadTemplateTypes === 'function') {
        loadTemplateTypes({ preserveSelection: true, suppressDefaultSelection: false });
    } else {
        window.__refreshTemplatesOnViewerReady = true;
    }
    window.dispatchEvent(new CustomEvent('idea-auth-ready'));
}

function completeSignOut() {
    console.log("User signed out");
    window.currentUser = null;
    idToken = null;
    window.__refreshTemplatesOnViewerReady = false;
    document.querySelector('.layout-container')?.setAttribute('inert', '');
    window.dispatchEvent(new CustomEvent('idea-auth-cleared'));

    // Show overlay
    loginOverlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Clear Header UI
    if (authContainer) authContainer.innerHTML = '';
}

function buildDevUser() {
    if (!devAuthConfig.token) {
        throw new Error('Local development auth is missing its bearer token');
    }
    return {
        uid: devAuthConfig.uid || 'local-dev-user',
        email: devAuthConfig.email || 'local-dev@example.local',
        displayName: devAuthConfig.displayName || 'Local Dev',
        photoURL: '',
        isAdmin: Boolean(devAuthConfig.isAdmin),
        getIdToken: async () => devAuthConfig.token,
    };
}

async function activateDevAuth() {
    console.log("Using local-dev-auth");
    await completeSignIn(buildDevUser());
}

if (isDevAuthEnabled) {
    activateDevAuth();
} else if (typeof firebase !== 'undefined' && firebase.auth) {
    firebase.auth().onAuthStateChanged(async (user) => {
        if (user) {
            await completeSignIn(user);
        } else {
            completeSignOut();
        }
    });
} else {
    console.error("Firebase Auth not loaded!");
}

// 6. UI Rendering
function renderUserUI(user) {
    if (!authContainer) return;

    const photoURL = getSafeUserImageUrl(user.photoURL);
    const name = String(user.displayName || user.email || 'Signed-in user');
    const email = String(user.email || '');

    authContainer.innerHTML = `
        <div class="dropdown">
            <button class="btn btn-link text-decoration-none dropdown-toggle d-flex align-items-center text-dark"
                    type="button" id="userDropdown" data-bs-toggle="dropdown" aria-expanded="false">
                ${photoURL
                    ? '<img alt="User" class="rounded-circle me-2 border user-avatar" style="width: 32px; height: 32px; object-fit: cover;">'
                    : '<span class="user-initials me-2" aria-hidden="true"></span>'}
                <span class="d-none d-md-block fw-bold small user-display-name"></span>
            </button>
            <ul class="dropdown-menu dropdown-menu-end shadow border-0" aria-labelledby="userDropdown">
                <li><h6 class="dropdown-header text-truncate user-email" style="max-width: 200px;"></h6></li>
                <li><hr class="dropdown-divider"></li>
                <li><a class="dropdown-item text-danger sign-out-link" href="#"><i class="fas fa-sign-out-alt me-2"></i>Sign Out</a></li>
            </ul>
        </div>
    `;
    if (photoURL) authContainer.querySelector('.user-avatar').src = photoURL;
    else authContainer.querySelector('.user-initials').textContent = name.trim().charAt(0).toUpperCase() || '?';
    authContainer.querySelector('.user-display-name').textContent = name;
    authContainer.querySelector('.user-email').textContent = email;
    authContainer.querySelector('.sign-out-link').addEventListener('click', event => {
        event.preventDefault();
        signOut();
    });

    window.signOut = signOut;
}

// 7. Monkey-patch Fetch to inject Token
const originalFetch = window.fetch;
window.fetch = async function (url, options = {}) {
    const isInternal = isInternalRequestUrl(url);
    const user = window.currentUser;
    if (!isInternal || !user) return originalFetch(url, options);

    const inherited = typeof Request !== 'undefined' && url instanceof Request ? url.headers : undefined;
    const headers = new Headers(options.headers === undefined ? inherited : options.headers);
    try {
        idToken = await user.getIdToken();
    } catch (error) {
        idToken = null;
        throw new Error('Your session could not be refreshed. Check your connection and sign in again.');
    }
    if (window.currentUser !== user) throw new Error('Your account changed. Please try again.');
    headers.set('Authorization', `Bearer ${idToken}`);
    // A Request body can only be consumed once. Keep a copy for the one auth retry.
    const retryRequest = typeof Request !== 'undefined' && url instanceof Request ? url.clone() : url;
    let response = await originalFetch(url, { ...options, headers });
    if (window.currentUser !== user) throw new Error('Your account changed. Please try again.');
    if (response.status === 401 && window.currentUser === user) {
        try {
            idToken = await user.getIdToken(true);
            if (window.currentUser !== user) return response;
            headers.set('Authorization', `Bearer ${idToken}`);
            response = await originalFetch(retryRequest, { ...options, headers });
        } catch (_) {
            // Preserve the authentication response if token refresh is unavailable.
        }
        if (response.status === 401) completeSignOut();
    }
    if (window.currentUser && window.currentUser !== user) throw new Error('Your account changed. Please try again.');
    return response;
};

// 8. Check API Key Status
let providerStatusCheckVersion = 0;
async function checkApiKeyStatus() {
    const version = ++providerStatusCheckVersion;
    const startButton = document.getElementById('startButton');
    const autoRatingButton = document.getElementById('startAutoRating');
    if (startButton) startButton.disabled = true;
    if (autoRatingButton && !(typeof autoRatingInProgress !== 'undefined' && autoRatingInProgress)) autoRatingButton.disabled = true;
    try {
        const data = window.IdeaWorkbench
            ? await window.IdeaWorkbench.loadProviderSettings()
            : await fetch('/api/settings/status').then(response => response.json());
        if (version !== providerStatusCheckVersion) return;
        const banner = document.getElementById('apiKeyAlert');
        const modelSelect = document.getElementById('modelType') || document.getElementById('modelSelect');
        const selectedProvider = window.IdeaWorkbench?.selectedProvider()
            || modelSelect?.selectedOptions?.[0]?.dataset?.provider || 'google_generative_ai';
        const providerName = selectedProvider === 'openai' ? 'OpenAI' : 'Gemini';
        const selectedKeyConfigured = data.api_keys?.[selectedProvider]?.configured
            ?? (typeof data.api_key_missing === 'boolean' && !data.api_key_missing);

        if (!selectedKeyConfigured) {
            if (banner) banner.style.display = 'block';
            if (startButton) {
                startButton.disabled = true;
                startButton.title = data.api_keys?.[selectedProvider]?.error_code === 'stored_key_unreadable'
                    ? `Your saved ${providerName} key cannot be opened. Enter a replacement in Settings.`
                    : `Please configure your ${providerName} API key in Settings to start.`;
            }
        } else {
            if (banner) banner.style.display = 'none';
            if (startButton) {
                if (!(typeof isEvolutionRunning !== 'undefined' && isEvolutionRunning)) startButton.disabled = false;
                startButton.title = "Start Evolution";
            }
        }
        if (autoRatingButton && !(typeof autoRatingInProgress !== 'undefined' && autoRatingInProgress)) {
            autoRatingButton.disabled = !selectedKeyConfigured;
        }
        window.apiKeyMissing = !selectedKeyConfigured;
        // Also update settings state for the selected provider.
        window.hasApiKey = selectedKeyConfigured;
        window.IdeaWorkbench?.updateProviderReadiness();

        // Admin Dashboard Link injection
        if (data.is_admin) {
            const dropdownMenu = document.querySelector('.dropdown-menu');
            if (dropdownMenu && !document.getElementById('adminLink')) {
                const adminLinkItem = document.createElement('li');
                adminLinkItem.innerHTML = `<a class="dropdown-item" href="/admin/" id="adminLink"><i class="fas fa-shield-alt me-2"></i>Admin Dashboard</a>`;
                // Insert before Sign Out (last item usually)
                dropdownMenu.insertBefore(adminLinkItem, dropdownMenu.lastElementChild);
                // Insert divider
                const divider = document.createElement('li');
                divider.innerHTML = '<hr class="dropdown-divider">';
                dropdownMenu.insertBefore(divider, dropdownMenu.lastElementChild);
            }
        }
    } catch (e) {
        if (version !== providerStatusCheckVersion) return;
        console.error("Failed to check API key status", e);
        window.apiKeyMissing = true;
        window.hasApiKey = false;
        const banner = document.getElementById('apiKeyAlert');
        if (banner) banner.style.display = 'block';
        if (startButton) {
            startButton.disabled = true;
            startButton.title = 'Could not check your provider key. Open Settings to retry.';
        }
        if (autoRatingButton && !(typeof autoRatingInProgress !== 'undefined' && autoRatingInProgress)) autoRatingButton.disabled = true;
    }
}

// Export for other modules if needed
window.checkApiKeyStatus = checkApiKeyStatus;
window.addEventListener('idea-provider-selection', () => void checkApiKeyStatus());
