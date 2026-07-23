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
    const fallback = '/static/img/logo.png';
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
        // Invalid or active URLs use the local inert fallback.
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
    loginOverlay.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.98); z-index: 10000;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        backdrop-filter: blur(10px);
    `;
    loginOverlay.innerHTML = `
        <div class="text-center p-5 bg-white shadow-lg rounded-3 border" style="max-width: 400px;">
            <img src="/static/img/logo.png" alt="Logo" style="height: 80px; margin-bottom: 25px;">
            <h2 class="mb-3 fw-bold">Idea Evolution</h2>
            <p class="text-muted mb-4">Sign in to access your evolution workspace.</p>
            <button id="googleSignInBtn" class="btn btn-primary btn-lg w-100 shadow-sm">
                <i class="fab fa-google me-2"></i> Sign in with Google
            </button>
            <div class="mt-3 text-muted small">
                By continuing, you agree to the Terms of Service.
            </div>
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

    console.log("Starting Google Sign-In...");
    const provider = new firebase.auth.GoogleAuthProvider();
    try {
        await firebase.auth().signInWithPopup(provider);
    } catch (error) {
        console.error("Login failed:", error);
        alert("Login failed: " + error.message);
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
    console.log("User signed in:", user.email);
    window.currentUser = user;
    try {
        idToken = await user.getIdToken();
    } catch (e) {
        console.error("Failed to get ID token", e);
    }

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
                <img alt="User" class="rounded-circle me-2 border user-avatar" style="width: 32px; height: 32px; object-fit: cover;">
                <span class="d-none d-md-block fw-bold small user-display-name"></span>
            </button>
            <ul class="dropdown-menu dropdown-menu-end shadow border-0" aria-labelledby="userDropdown">
                <li><h6 class="dropdown-header text-truncate user-email" style="max-width: 200px;"></h6></li>
                <li><hr class="dropdown-divider"></li>
                <li><a class="dropdown-item text-danger sign-out-link" href="#"><i class="fas fa-sign-out-alt me-2"></i>Sign Out</a></li>
            </ul>
        </div>
    `;
    authContainer.querySelector('.user-avatar').src = photoURL;
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
    // Only same-origin requests may receive the bearer token. Substring checks
    // are unsafe because an attacker URL can include our origin in its path.
    const isInternal = isInternalRequestUrl(url);

    if (isInternal && window.currentUser) {
        // Refresh token if potentially expired? (Firebase SDK handles internal caching, getIdToken() manages refresh)
        try {
            // true = force refresh, false = use cached.
            // We use cached unless we get 401? For now, just getting it ensures validity.
            idToken = await window.currentUser.getIdToken();
        } catch (e) {
            console.warn("Token fetch error:", e);
        }

        if (idToken) {
            if (!options.headers) {
                options.headers = {};
            }

            // Handle Headers object vs plain object
            if (options.headers instanceof Headers) {
                options.headers.append('Authorization', `Bearer ${idToken}`);
            } else {
                options.headers['Authorization'] = `Bearer ${idToken}`;
            }
        }
    }

    // Call original
    const response = await originalFetch(url, options);

    // Handle 401 Unauthorized globally
    if (response.status === 401) {
        console.warn("Resource Unauthorized. Checking auth state...");
        // If we get 401, maybe token expired and sdk didn't refresh?
        // Or user deleted?
    }

    return response;
};

// 8. Check API Key Status
async function checkApiKeyStatus() {
    try {
        const response = await fetch('/api/settings/status');
        const data = await response.json();
        const banner = document.getElementById('apiKeyAlert');
        const startButton = document.getElementById('startButton');

        if (data.api_key_missing) {
            if (banner) banner.style.display = 'block';
            if (startButton) {
                startButton.disabled = true;
                startButton.title = "Please configure your Gemini API Key in Settings to start.";
            }
        } else {
            if (banner) banner.style.display = 'none';
            if (startButton) {
                startButton.disabled = false;
                startButton.title = "Start Evolution";
            }
        }
        // Also update settings modal state if needed
        window.hasApiKey = !data.api_key_missing;

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
        console.error("Failed to check API key status", e);
    }
}

// Export for other modules if needed
window.checkApiKeyStatus = checkApiKeyStatus;
