/* Shared workspace behavior: one responsive sidebar and one candidate inspector. */
(function () {
    'use strict';
    const narrow = window.matchMedia('(max-width: 768px)');
    let inspector = null;
    let focusOrigin = null;
    let desktopCollapsed = false;
    let sidebarOrigin = null;
    let providerSettings = null;
    let providerSettingsOwner = null;
    let providerSettingsRequest = null;
    let providerSettingsError = null;
    let providerSettingsViewRequest = 0;
    let sidebarViewOrigin = null;
    let billingRequestId = 0;
    let closeControlHelp = null;
    const modelPickers = new Map();

    function providerName(provider) {
        if (provider === 'openai') return 'OpenAI';
        if (provider === 'google_generative_ai' || provider === 'gemini') return 'Gemini';
        if (provider === 'ollama') return 'Ollama (local)';
        return provider || 'Unknown provider';
    }

    function modelProvider(model) {
        return model?.provider === 'openai' || String(model?.id || '').startsWith('gpt-')
            ? 'openai' : 'google_generative_ai';
    }

    function preferenceKey(provider) {
        return `idea.lastModel.${window.currentUser?.uid || 'local'}.${provider}`;
    }

    function preferredModel(models, provider, { requested = null, remembered = null, defaultModel = null } = {}) {
        const available = models.filter(model => modelProvider(model) === provider);
        for (const id of [requested, remembered, defaultModel]) {
            if (available.some(model => model.id === id)) return id;
        }
        return available.find(model => model.recommended || model.is_default)?.id || available[0]?.id || '';
    }

    function selectedProvider() {
        const picker = document.getElementById('providerSelect') || document.getElementById('ratingProviderSelect');
        if (picker?.value) return picker.value;
        const model = document.getElementById('modelType') || document.getElementById('modelSelect');
        return model?.selectedOptions?.[0]?.dataset?.provider || 'google_generative_ai';
    }

    function setProviderSettings(data) {
        providerSettings = data;
        providerSettingsOwner = window.currentUser?.uid || 'local';
        providerSettingsError = null;
        updateProviderReadiness();
        window.updateTemplateGenerationReadiness?.();
        return data;
    }

    async function loadProviderSettings(force = false) {
        const owner = window.currentUser?.uid || 'local';
        if (providerSettingsOwner !== owner) {
            providerSettings = null;
            providerSettingsError = null;
        }
        if (providerSettings && !force) return providerSettings;
        if (providerSettingsRequest?.owner === owner && !force) return providerSettingsRequest.promise;
        const request = { owner };
        providerSettingsRequest = request;
        request.promise = fetch('/api/settings/status').then(async response => {
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || 'Could not check provider keys');
            if (owner !== (window.currentUser?.uid || 'local') || providerSettingsRequest !== request) {
                throw new Error('Your session changed. Open provider settings again to refresh.');
            }
            return setProviderSettings(data);
        }).catch(error => {
            if (providerSettingsRequest === request && owner === (window.currentUser?.uid || 'local')) {
                providerSettingsError = error;
                updateProviderReadiness();
            }
            throw error;
        }).finally(() => {
            if (providerSettingsRequest === request) providerSettingsRequest = null;
        });
        return request.promise;
    }

    function updateProviderReadiness() {
        const provider = selectedProvider();
        const name = providerName(provider);
        const configured = providerSettings?.api_keys?.[provider]?.configured;
        const unreadable = providerSettings?.api_keys?.[provider]?.error_code === 'stored_key_unreadable';
        const status = document.getElementById('selectedProviderStatus');
        const setup = document.getElementById('configureSelectedProvider');
        if (status) {
            status.textContent = providerSettingsError ? 'Could not check provider key' : unreadable ? `${name} saved key unreadable` : configured === true ? `${name} key ready` : configured === false ? `${name} key needed` : 'Checking provider key…';
            status.dataset.state = providerSettingsError || unreadable ? 'error' : configured === true ? 'ready' : configured === false ? 'missing' : 'checking';
        }
        if (setup) {
            setup.hidden = configured !== false && !providerSettingsError;
            setup.textContent = providerSettingsError ? 'Check settings' : unreadable ? `Replace ${name} key` : `Add ${name} key`;
        }
    }

    function configureModelPicker({ modelId, providerId, models, defaultModel }) {
        const modelSelect = document.getElementById(modelId);
        const providerSelect = document.getElementById(providerId);
        if (!modelSelect || !providerSelect || !Array.isArray(models)) return;
        const catalog = models.map(model => ({ ...model, provider: modelProvider(model) }));
        const picker = { catalog, defaultModel, modelSelect, providerSelect };
        modelPickers.set(modelId, picker);
        let provider = catalog.find(model => model.id === modelSelect.value)?.provider
            || catalog.find(model => model.id === defaultModel)?.provider || 'google_generative_ai';
        try {
            const remembered = localStorage.getItem(`idea.selectedProvider.${window.currentUser?.uid || 'local'}`);
            if (catalog.some(model => model.provider === remembered)) provider = remembered;
        } catch (_) { /* model preferences are optional */ }

        const applyProvider = (nextProvider, requested = null) => {
            const available = catalog.filter(model => model.provider === nextProvider);
            let remembered = null;
            try { remembered = localStorage.getItem(preferenceKey(nextProvider)); } catch (_) { /* optional */ }
            const selected = preferredModel(catalog, nextProvider, { requested, remembered, defaultModel });
            providerSelect.value = nextProvider;
            modelSelect.replaceChildren();
            const group = document.createElement('optgroup');
            group.label = `${providerName(nextProvider)} models`;
            available.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name || model.id;
                option.dataset.provider = nextProvider;
                option.selected = model.id === selected;
                group.appendChild(option);
            });
            modelSelect.appendChild(group);
            modelSelect.disabled = available.length === 0;
            modelSelect.dispatchEvent(new Event('change', { bubbles: true }));
        };
        picker.applyProvider = applyProvider;
        if (providerSelect._modelPickerChange) providerSelect.removeEventListener('change', providerSelect._modelPickerChange);
        providerSelect._modelPickerChange = () => applyProvider(providerSelect.value);
        providerSelect.addEventListener('change', providerSelect._modelPickerChange);
        if (!modelSelect.dataset.preferenceBound) {
            modelSelect.dataset.preferenceBound = 'true';
            modelSelect.addEventListener('change', () => {
                const current = modelPickers.get(modelId);
                const activeProvider = current?.providerSelect.value || selectedProvider();
                try {
                    localStorage.setItem(preferenceKey(activeProvider), modelSelect.value);
                    localStorage.setItem(`idea.selectedProvider.${window.currentUser?.uid || 'local'}`, activeProvider);
                } catch (_) { /* optional */ }
                updateProviderReadiness();
                window.dispatchEvent(new CustomEvent('idea-provider-selection', { detail: { provider: activeProvider, model: modelSelect.value } }));
            });
        }
        applyProvider(provider);
    }

    function selectModel(modelId, requested, { fallback = false } = {}) {
        const picker = modelPickers.get(modelId);
        const model = picker?.catalog.find(value => value.id === requested);
        if (!model) {
            if (fallback && picker && requested) {
                const provider = modelProvider({ id: requested });
                if (picker.catalog.some(value => value.provider === provider)) picker.applyProvider(provider);
            }
            return false;
        }
        picker.applyProvider(model.provider, requested);
        return true;
    }

    function openProviderSettings(providerOverride = null) {
        const request = ++providerSettingsViewRequest;
        const owner = window.currentUser?.uid || 'local';
        const provider = typeof providerOverride === 'string' ? providerOverride : selectedProvider();
        const selector = document.getElementById('apiKeyProviderSelect');
        if (selector) {
            selector.value = provider;
            const input = document.getElementById('apiKeyInput');
            if (input) input.value = '';
        }
        openSidebar();
        if (typeof window.showSidebarView === 'function') {
            Promise.resolve(window.showSidebarView('settings')).then(() => {
                focusProviderSettings(request, owner);
            });
        }
        else {
            selectSidebarView('settings');
            const save = document.getElementById('saveSettingsBtn');
            document.getElementById('apiKeyInputContainer')?.style.setProperty('display', 'none');
            document.getElementById('apiKeyConfiguredContainer')?.style.setProperty('display', 'none');
            if (selector) selector.disabled = true;
            if (save) save.disabled = true;
            showSharedKeyError(new Error('Checking saved provider keys…'), 'info');
            void loadProviderSettings(true).then(data => {
                if (request !== providerSettingsViewRequest || owner !== (window.currentUser?.uid || 'local')) return;
                clearSharedKeyStatus();
                renderSharedKeySettings(data);
            }).catch(error => {
                if (request !== providerSettingsViewRequest || owner !== (window.currentUser?.uid || 'local')) return;
                document.getElementById('apiKeyInputContainer')?.style.setProperty('display', 'block');
                showSharedKeyError(new Error('Could not check saved keys. Reopen provider settings to retry, or save a replacement key.'));
            }).finally(() => {
                if (request !== providerSettingsViewRequest) return;
                if (selector) selector.disabled = false;
                if (save) save.disabled = false;
                focusProviderSettings(request, owner);
            });
        }
        const settings = document.getElementById('view-settings');
        if (settings) settings.scrollTop = 0;
    }

    function focusProviderSettings(request, owner) {
        const settings = document.getElementById('view-settings');
        const panel = document.getElementById('controlsPanel');
        // A late status response must not reopen the drawer or steal focus.
        if (request !== providerSettingsViewRequest || owner !== (window.currentUser?.uid || 'local')
            || !settings?.classList.contains('active') || panel?.hasAttribute('inert')
            || document.activeElement !== settings) return;
        const target = ['apiKeyInput', 'replaceApiKeyBtn', 'apiKeyProviderSelect']
            .map(id => document.getElementById(id))
            .find(node => node && !node.disabled && node.offsetParent !== null);
        target?.focus({ preventScroll: true });
        target?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }

    function renderSharedKeySettings(data) {
        const provider = document.getElementById('apiKeyProviderSelect')?.value || selectedProvider();
        const status = data?.api_keys?.[provider];
        const input = document.getElementById('apiKeyInputContainer');
        const configured = document.getElementById('apiKeyConfiguredContainer');
        if (input) input.style.display = status?.configured ? 'none' : 'block';
        if (configured) configured.style.display = status?.configured ? 'block' : 'none';
        const label = document.getElementById('apiKeyProviderLabel');
        if (label) label.textContent = `${providerName(provider)} API key`;
        const name = document.getElementById('configuredProviderName');
        if (name) name.textContent = providerName(provider);
        const masked = document.getElementById('maskedKeyDisplay');
        if (masked) masked.textContent = status?.masked_key || '••••';
        const source = status?.source || data?.key_sources?.[provider];
        const sourceNote = document.getElementById('apiKeySourceNote');
        if (sourceNote) sourceNote.textContent = source === 'environment'
            ? 'Using this machine’s local environment key. Save a replacement only if you want to override it.'
            : 'Saved securely for this provider. Switching models reuses this key.';
        const remove = document.getElementById('deleteApiKeyBtn');
        if (remove) remove.hidden = source === 'environment';
        const cancel = document.getElementById('cancelApiKeyEdit');
        if (cancel) cancel.hidden = true;
        if (status?.error_code === 'stored_key_unreadable') {
            showSharedKeyError(new Error(status.message || 'This saved key cannot be opened. Enter a replacement key to use this provider.'), 'warning');
        }
    }

    function editProviderKey() {
        const input = document.getElementById('apiKeyInput');
        if (input) { input.value = ''; input.removeAttribute('aria-invalid'); }
        clearSharedKeyStatus();
        document.getElementById('apiKeyConfiguredContainer')?.style.setProperty('display', 'none');
        document.getElementById('apiKeyInputContainer')?.style.setProperty('display', 'block');
        const cancel = document.getElementById('cancelApiKeyEdit');
        if (cancel) cancel.hidden = false;
        input?.focus();
    }

    function cancelProviderKeyEdit() {
        const input = document.getElementById('apiKeyInput');
        if (input) input.value = '';
        clearSharedKeyStatus();
        renderSharedKeySettings(providerSettings);
        document.getElementById('replaceApiKeyBtn')?.focus();
    }

    function clearSharedKeyStatus() {
        const message = document.getElementById('settingsStatus');
        if (message) { message.hidden = true; message.textContent = ''; }
    }

    function showSharedKeyError(error, type = 'danger') {
        const message = document.getElementById('settingsStatus');
        if (message) {
            message.hidden = false;
            message.style.display = 'block';
            message.className = `alert alert-${type} mt-3`;
            message.textContent = error.message || String(error);
        }
    }

    async function saveSharedKey() {
        const input = document.getElementById('apiKeyInput');
        if (!input?.value.trim()) {
            input?.setAttribute('aria-invalid', 'true');
            input?.focus();
            return showSharedKeyError(new Error('Enter a key for this provider.'));
        }
        const button = document.getElementById('saveSettingsBtn');
        if (button.disabled) return;
        const selector = document.getElementById('apiKeyProviderSelect');
        const provider = selector.value;
        const label = button.textContent;
        const cancel = document.getElementById('cancelApiKeyEdit');
        const owner = window.currentUser?.uid || 'local';
        button.disabled = true;
        button.textContent = 'Saving…';
        selector.disabled = true;
        input.disabled = true;
        input.removeAttribute('aria-invalid');
        if (cancel) cancel.disabled = true;
        clearSharedKeyStatus();
        try {
            const response = await fetch('/api/settings/api-key', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider, api_key: input.value.trim() }) });
            const data = await response.json();
            if (!response.ok || data.status === 'error') throw new Error(data.message || 'Could not save this key');
            input.value = '';
            if (owner !== (window.currentUser?.uid || 'local')) return;
            try {
                const status = await loadProviderSettings(true);
                renderSharedKeySettings(status);
                await window.checkApiKeyStatus?.();
                showSharedKeyError(new Error(`${providerName(provider)} key saved. You can use this provider in your next run.`), 'success');
            } catch (_) {
                if (owner !== (window.currentUser?.uid || 'local')) return;
                showSharedKeyError(new Error(`${providerName(provider)} key was saved, but its status could not be refreshed. Reopen provider settings to check it.`), 'warning');
            }
        } catch (error) {
            if (owner === (window.currentUser?.uid || 'local')) showSharedKeyError(error);
            else input.value = '';
        } finally {
            button.disabled = false;
            button.textContent = label;
            selector.disabled = false;
            input.disabled = false;
            if (cancel) cancel.disabled = false;
        }
    }

    function money(value) {
        const number = finite(value);
        return number === null ? 'Not recorded' : new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(number);
    }

    function billingCoverageText(coverage, count = 0) {
        const gaps = [];
        if (coverage?.runs_missing_cost) gaps.push(`${coverage.runs_missing_cost} ${coverage.runs_missing_cost === 1 ? 'run has no recorded cost and is' : 'runs have no recorded cost and are'} excluded from the total`);
        if (coverage?.runs_missing_usage) gaps.push(`${coverage.runs_missing_usage} ${coverage.runs_missing_usage === 1 ? 'run has' : 'runs have'} incomplete token details`);
        if (coverage?.missing_usage_requests) gaps.push(`${coverage.missing_usage_requests} ${coverage.missing_usage_requests === 1 ? 'request returned' : 'requests returned'} no usage; their spend is unknown`);
        if (coverage?.legacy_price_runs) gaps.push(`${coverage.legacy_price_runs} older ${coverage.legacy_price_runs === 1 ? 'run uses a legacy price estimate' : 'runs use legacy price estimates'}`);
        if (coverage?.estimated_input_tokens) gaps.push(`${Number(coverage.estimated_input_tokens).toLocaleString()} input tokens were estimated locally`);
        if (coverage?.unpriced_models?.length) gaps.push(`Pricing unavailable for ${coverage.unpriced_models.join(', ')}`);
        if (coverage?.truncated) gaps.push('Only part of the saved history is included');
        const recorded = coverage?.runs_with_recorded_cost ?? count;
        return gaps.length ? `${gaps.join('. ')}.` : `${recorded} ${recorded === 1 ? 'run has' : 'runs have'} recorded cost. Provider invoices may include other activity.`;
    }

    function appendText(parent, tag, text, className = '') {
        const element = document.createElement(tag);
        element.textContent = String(text);
        if (className) element.className = className;
        parent.appendChild(element);
        return element;
    }

    function appendUsage(parent, data) {
        const list = document.createElement('dl');
        list.className = 'billing-usage';
        [['Input tokens', data.input_tokens], ['Output tokens', data.output_tokens], ['Thinking tokens', data.thought_tokens], ['Cache reads (tokens)', data.cached_input_tokens], ['Cache writes (tokens)', data.cache_write_input_tokens]].forEach(([label, value]) => {
            appendText(list, 'dt', label);
            appendText(list, 'dd', finite(value) === null ? 'Not recorded' : Number(value).toLocaleString());
        });
        parent.appendChild(list);
    }

    function renderBilling(data) {
        const container = document.getElementById('billingOverview');
        if (!container) return;
        container.replaceChildren();
        const totals = data.totals || {};
        appendText(container, 'p', data.coverage?.truncated ? 'Loaded recorded runs · recorded estimate' : 'All recorded runs · recorded estimate', 'billing-kicker');
        appendText(container, 'strong', money(totals.estimated_cost), 'billing-total');
        appendText(container, 'p', `${totals.run_count ?? 0} runs · USD · Includes recorded evolution and re-rating costs.`, 'billing-description');
        appendText(container, 'p', billingCoverageText(data.coverage, totals.run_count), 'billing-coverage');
        appendUsage(container, totals);
        appendText(container, 'p', `Estimated cache savings: ${money(totals.cache_savings)} compared with uncached input pricing.`, 'billing-description');
        appendText(container, 'p', 'These are estimates from recorded usage, not your provider invoice. Stored run prices are preserved where available.', 'billing-description');
        [['By provider', data.by_provider, 'provider'], ['By model', data.by_model, 'model']].forEach(([title, rows, key]) => {
            if (!Array.isArray(rows) || !rows.length) return;
            appendText(container, 'h3', title, 'billing-section-title');
            rows.forEach(row => {
                const section = document.createElement('details');
                section.className = 'billing-item';
                const label = key === 'provider' ? providerName(row.provider) : (row.model || row.model_name || row.id || 'Unknown model');
                appendText(section, 'summary', `${label} · ${money(row.estimated_cost)}`);
                appendUsage(section, row);
                appendText(section, 'p', `Cache savings: ${money(row.cache_savings)}`, 'billing-description');
                container.appendChild(section);
            });
        });
        appendText(container, 'h3', 'Previous runs', 'billing-section-title');
        if (!data.runs?.length) appendText(container, 'p', 'Your recorded runs will appear here as you experiment.', 'billing-description');
        (data.runs || []).forEach(run => {
            const item = document.createElement('details');
            item.className = 'billing-item';
            appendText(item, 'summary', `${run.name || run.id || 'Untitled run'} · ${money(run.estimated_cost)}`);
            appendText(item, 'p', `${run.status || 'Saved'}${run.updated_at ? ` · ${new Date(run.updated_at).toLocaleDateString()}` : ''}`, 'billing-description');
            appendUsage(item, run);
            appendText(item, 'p', `Evolution: ${money(run.evolution_cost)} · Re-rating: ${money(run.rerating_cost)} · Cache savings: ${money(run.cache_savings)}`, 'billing-description');
            (run.models || []).forEach(model => appendText(item, 'p', `${model.model || model.model_name || model.id || 'Unknown model'} · ${money(model.estimated_cost)}`, 'billing-description'));
            container.appendChild(item);
        });
        (data.notes || []).forEach(note => appendText(container, 'p', note, 'billing-description'));
        const links = document.createElement('div');
        links.className = 'billing-dashboard-links';
        (data.dashboards || []).forEach(dashboard => {
            try {
                const url = new URL(dashboard.url);
                if (url.protocol !== 'https:') return;
                const link = appendText(links, 'a', dashboard.label || `${providerName(dashboard.provider)} usage`);
                link.href = url.href;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
            } catch (_) { /* malformed links stay inert */ }
        });
        container.appendChild(links);
    }

    async function showCosts() {
        selectSidebarView('costs');
        const container = document.getElementById('billingOverview');
        if (!container) return;
        const requestId = ++billingRequestId;
        const owner = window.currentUser?.uid || 'local';
        const status = document.getElementById('billingStatus');
        container.setAttribute('aria-busy', 'true');
        if (status) { status.hidden = false; status.textContent = 'Loading recorded costs…'; }
        try {
            const response = await fetch('/api/billing');
            const data = await response.json();
            if (requestId !== billingRequestId || owner !== (window.currentUser?.uid || 'local')) return;
            if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Could not load recorded costs');
            renderBilling(data);
            if (status) { status.textContent = 'Recorded costs updated.'; status.hidden = true; }
        } catch (error) {
            if (requestId !== billingRequestId || owner !== (window.currentUser?.uid || 'local')) return;
            if (status) { status.hidden = false; status.textContent = error.message; }
            else container.textContent = error.message;
        } finally {
            if (requestId === billingRequestId) container.setAttribute('aria-busy', 'false');
        }
    }

    function finite(value) {
        if (value === null || value === undefined || value === '') return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function historySamplingNotice(history) {
        const rows = Array.isArray(history) ? history : [history];
        const row = [...rows].reverse().find(value => value?.history_sampling || value?.metadata?.history_sampling || value?.sample_size !== undefined);
        if (!row || (row.embedding_scope || row.metadata?.embedding_scope) === 'current_generation') return '';
        const sample = row.history_sampling || row.metadata?.history_sampling || row;
        if (sample.truncated !== true) return '';
        const sampled = finite(sample.sample_size);
        const total = finite(sample.total_size);
        const slots = finite(sample.total_history_slots);
        const size = sampled !== null && total !== null
            ? `${sampled.toLocaleString()} of ${total.toLocaleString()} unique revisions`
            : 'a bounded subset of unique revisions';
        return `Historical diversity is sampled: ${size}${slots !== null ? ` (${slots.toLocaleString()} recorded history slots)` : ''}. Current population and island metrics remain exact.`;
    }

    function renderHistorySamplingNote(container, history) {
        const notice = historySamplingNotice(history);
        if (container) {
            container.textContent = notice;
            container.hidden = !notice;
        }
        return notice;
    }

    function metricEvidence(idea, suite = {}) {
        const records = idea?.evaluations || {};
        const contributions = idea?.selection_metrics?.components || {};
        const configured = suite?.fitness?.metrics || [];
        const metrics = configured.length ? configured : Object.entries(records).flatMap(([id, record]) =>
            Object.keys(record?.metrics || {}).map(name => ({ component_id: id, metric_name: name }))
        );
        return metrics.map(metric => {
            const key = `${metric.component_id}.${metric.metric_name}`;
            const record = records[metric.component_id];
            const contribution = contributions[key];
            const status = record?.status || contribution?.evaluation_status || 'missing';
            const valid = ['ok', 'cached'].includes(status);
            return {
                key, name: metric.metric_name.replace(/_/g, ' '), evaluator: metric.component_id,
                value: valid ? finite(record?.metrics?.[metric.metric_name] ?? contribution?.raw_value) : null,
                normalized: valid ? finite(record?.normalized_metrics?.[metric.metric_name] ?? contribution?.score_value) : null,
                weight: finite(metric.weight ?? contribution?.weight), status,
                direction: metric.direction || 'maximize', normalization: metric.normalization || null,
                reason: record?.error || contribution?.reason || null,
                version: record?.evaluator_version || record?.component_version || record?.schema_version || null,
            };
        });
    }

    function renderMetrics(container, idea, suite) {
        if (!container) return;
        container.replaceChildren();
        const evidence = metricEvidence(idea, suite);
        container.hidden = evidence.length === 0 && finite(idea?.selection_metrics?.fitness) === null;
        if (container.hidden) return;
        const aggregate = finite(idea?.selection_metrics?.fitness);
        if (aggregate !== null) {
            const summary = document.createElement('p');
            summary.className = 'inspector-fitness';
            summary.textContent = `Selection fitness ${aggregate.toFixed(4)}`;
            container.appendChild(summary);
        }
        evidence.forEach(metric => {
            const row = document.createElement('div');
            row.className = 'inspector-metric';
            const title = document.createElement('strong');
            title.textContent = metric.name;
            const value = document.createElement('span');
            value.textContent = metric.value === null ? metric.status : metric.value.toLocaleString(undefined, { maximumFractionDigits: 4 });
            const description = document.createElement('small');
            const weight = metric.weight === null ? 'weight unspecified' : `weight ${metric.weight}`;
            description.textContent = `${metric.evaluator} · ${metric.status} · ${metric.direction === 'maximize' ? 'higher is better' : metric.direction} · ${weight}${metric.normalization ? ` · ${metric.normalization.replace(/_/g, ' ')}` : ''}`;
            row.append(title, value, description);
            if (metric.reason) {
                const reason = document.createElement('small');
                reason.textContent = metric.reason;
                row.appendChild(reason);
            }
            container.appendChild(row);
        });
    }

    async function renderPrimaryArtifact(container, idea, evolutionId) {
        if (!container) return;
        container.replaceChildren();
        const refs = idea?.artifact_refs || idea?.idea?.artifact_refs || [];
        const ref = Array.isArray(refs) ? refs.find(value => value?.role === 'primary' && value?.media_type === 'image/png' && value?.id) : null;
        container.hidden = !ref || !evolutionId;
        if (container.hidden) return;
        const figure = document.createElement('figure');
        figure.className = 'idea-artifact-figure';
        const image = document.createElement('img');
        image.alt = `Visualization for ${idea?.title || idea?.idea?.title || 'candidate'}`;
        const caption = document.createElement('figcaption');
        caption.textContent = 'Primary visualization';
        figure.append(image, caption);
        container.appendChild(figure);
        try {
            const response = await fetch(`/api/evolution/${encodeURIComponent(evolutionId)}/artifacts/${encodeURIComponent(ref.id)}`, { headers: { Accept: 'image/png' } });
            if (!response.ok) throw new Error(`request failed (${response.status})`);
            const blob = await response.blob();
            if (blob.type && blob.type !== 'image/png') throw new Error('artifact is not a PNG');
            if (!image.isConnected) return;
            const url = URL.createObjectURL(blob);
            image.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
            image.addEventListener('error', () => URL.revokeObjectURL(url), { once: true });
            image.src = url;
        } catch (error) {
            image.hidden = true;
            caption.textContent = `Visualization unavailable: ${error.message}`;
        }
    }

    function focusable(element) {
        return [...element.querySelectorAll('button, a[href], input, select, textarea, [tabindex="0"]')]
            .filter(node => !node.disabled && !node.closest('[hidden], [inert]') && node.getClientRects().length);
    }

    function setupControlHelp() {
        document.querySelectorAll('[data-control-help]').forEach(button => {
            const content = document.getElementById(button.dataset.controlHelp);
            const row = button.closest('.control-label-row');
            if (!content || !row) return;
            let pinned = false;
            const hide = () => {
                pinned = false;
                content.hidden = true;
                button.setAttribute('aria-expanded', 'false');
                if (closeControlHelp === hide) closeControlHelp = null;
            };
            const show = () => {
                if (closeControlHelp !== hide) closeControlHelp?.();
                closeControlHelp = hide;
                content.hidden = false;
                button.setAttribute('aria-expanded', 'true');
            };
            button.addEventListener('pointerenter', show);
            button.addEventListener('focus', show);
            button.addEventListener('blur', () => { if (!pinned) hide(); });
            row.addEventListener('pointerleave', () => {
                if (!pinned && document.activeElement !== button) hide();
            });
            button.addEventListener('click', () => {
                pinned = !pinned;
                if (pinned) show(); else hide();
            });
            button.addEventListener('keydown', event => {
                if (event.key !== 'Escape' || content.hidden) return;
                event.preventDefault();
                event.stopPropagation();
                hide();
            });
            document.addEventListener('click', event => {
                if (!content.hidden && !row.contains(event.target)) hide();
            });
        });
    }

    function syncInspector() {
        if (!inspector || inspector.hidden) return;
        inspector.setAttribute('role', narrow.matches ? 'dialog' : 'complementary');
        if (narrow.matches) inspector.setAttribute('aria-modal', 'true');
        else inspector.removeAttribute('aria-modal');
        document.querySelector('.layout-container')?.toggleAttribute('inert', narrow.matches);
        document.body.classList.toggle('inspector-mobile-open', narrow.matches);
    }

    function openInspector(element) {
        if (!element) return;
        const needsFocus = !inspector || inspector.hidden || !element.contains(document.activeElement);
        if (!inspector || inspector.hidden) focusOrigin = document.activeElement;
        inspector = element;
        inspector.hidden = false;
        inspector.removeAttribute('aria-hidden');
        document.body.classList.add('inspector-open');
        syncInspector();
        if (needsFocus) element.querySelector('[data-inspector-close]')?.focus({ preventScroll: true });
    }

    function closeInspector() {
        if (!inspector || inspector.hidden) return;
        inspector.hidden = true;
        inspector.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('inspector-open', 'inspector-mobile-open');
        document.querySelector('.layout-container')?.removeAttribute('inert');
        inspector.dispatchEvent(new Event('hidden.bs.modal'));
        if (focusOrigin?.isConnected && !focusOrigin.closest('[hidden], [inert]')) focusOrigin.focus({ preventScroll: true });
        focusOrigin = null;
    }

    function syncSidebar() {
        const panel = document.getElementById('controlsPanel');
        const toggle = document.getElementById('sidebarToggle');
        if (!panel || !toggle) return;
        panel.classList.toggle('collapsed', !narrow.matches && desktopCollapsed);
        const open = narrow.matches ? panel.classList.contains('show') : !desktopCollapsed;
        panel.toggleAttribute('inert', !open);
        panel.setAttribute('aria-hidden', String(!open));
        toggle.setAttribute('aria-expanded', String(open));
        toggle.setAttribute('aria-label', open ? 'Close experiment setup' : 'Open experiment setup');
        const icon = toggle.querySelector('i');
        if (icon) icon.className = narrow.matches ? (open ? 'fas fa-times' : 'fas fa-bars') : (open ? 'fas fa-chevron-left' : 'fas fa-chevron-right');
        document.querySelectorAll('.main-content, .evolution-content').forEach(node => node.classList.toggle('sidebar-collapsed', !narrow.matches && desktopCollapsed));
        const backdrop = document.getElementById('sidebarBackdrop');
        if (backdrop) backdrop.hidden = !narrow.matches || !open;
        document.body.classList.toggle('sidebar-mobile-open', narrow.matches && open);
        if (narrow.matches) {
            document.querySelectorAll('.main-content, .evolution-content').forEach(node => node.toggleAttribute('inert', open));
        } else document.querySelectorAll('.main-content, .evolution-content').forEach(node => node.removeAttribute('inert'));
    }

    function closeSidebar() {
        document.getElementById('controlsPanel')?.classList.remove('show');
        syncSidebar();
        if (sidebarOrigin?.isConnected) sidebarOrigin.focus({ preventScroll: true });
    }

    function openSidebar() {
        const panel = document.getElementById('controlsPanel');
        if (!panel) return;
        const wasClosed = narrow.matches ? !panel.classList.contains('show') : desktopCollapsed;
        if (wasClosed) sidebarOrigin = document.activeElement;
        if (narrow.matches) panel.classList.add('show');
        else {
            desktopCollapsed = false;
            try { localStorage.setItem('sidebarCollapsed', 'false'); } catch (_) { /* optional */ }
        }
        syncSidebar();
    }

    function selectSidebarView(name, { focus = true } = {}) {
        const selected = document.getElementById(`view-${name}`);
        if (!selected) return;
        closeControlHelp?.();
        const previous = document.querySelector('.sidebar-view.active');
        if (previous?.id === 'view-main' && name !== 'main') sidebarViewOrigin = document.activeElement;
        document.querySelectorAll('.sidebar-view').forEach(view => {
            const active = view === selected;
            view.classList.toggle('active', active);
            view.classList.remove('slide-left');
            view.hidden = !active;
            view.toggleAttribute('inert', !active);
            view.setAttribute('aria-hidden', String(!active));
        });
        document.getElementById('controlsPanel')?.classList.toggle('expanded', ['history', 'costs', 'advanced', 'settings'].includes(name));
        if (focus) {
            const origin = name === 'main' && sidebarViewOrigin?.isConnected && selected.contains(sidebarViewOrigin)
                ? sidebarViewOrigin : null;
            if (origin) origin.focus({ preventScroll: true });
            else {
                selected.tabIndex = -1;
                selected.focus({ preventScroll: true });
            }
            if (name === 'main') sidebarViewOrigin = null;
        }
    }

    function renderChartTable(container, labels, datasets, caption = 'Chart data') {
        if (!container) return;
        container.replaceChildren();
        const table = document.createElement('table');
        table.className = 'chart-data-table';
        const title = table.createCaption();
        title.textContent = caption;
        const heading = table.createTHead().insertRow();
        ['Generation / idea', ...datasets.map(dataset => dataset.label)].forEach(label => {
            const cell = document.createElement('th');
            cell.scope = 'col';
            cell.textContent = label;
            heading.appendChild(cell);
        });
        const body = table.createTBody();
        labels.forEach((label, index) => {
            const row = body.insertRow();
            [label, ...datasets.map(dataset => {
                const number = finite(dataset.data?.[index]);
                return number === null ? 'Unavailable' : number.toLocaleString(undefined, { maximumFractionDigits: 4 });
            })].forEach((value, column) => {
                const cell = document.createElement(column === 0 ? 'th' : 'td');
                if (column === 0) cell.scope = 'row';
                cell.textContent = String(value);
                row.appendChild(cell);
            });
        });
        container.appendChild(table);
    }

    document.addEventListener('DOMContentLoaded', () => {
        try { desktopCollapsed = localStorage.getItem('sidebarCollapsed') === 'true'; } catch (_) { /* storage is optional */ }
        selectSidebarView('main', { focus: false });
        syncSidebar();
        setupControlHelp();
        document.getElementById('sidebarToggle')?.addEventListener('click', event => {
            sidebarOrigin = event.currentTarget;
            const panel = document.getElementById('controlsPanel');
            if (narrow.matches) panel?.classList.toggle('show');
            else {
                desktopCollapsed = !desktopCollapsed;
                try { localStorage.setItem('sidebarCollapsed', String(desktopCollapsed)); } catch (_) { /* optional */ }
            }
            syncSidebar();
            if (narrow.matches && panel?.classList.contains('show')) focusable(panel)[0]?.focus();
        });
        document.getElementById('sidebarBackdrop')?.addEventListener('click', closeSidebar);
        document.getElementById('configureSelectedProvider')?.addEventListener('click', openProviderSettings);
        document.getElementById('replaceApiKeyBtn')?.addEventListener('click', editProviderKey);
        document.getElementById('cancelApiKeyEdit')?.addEventListener('click', cancelProviderKeyEdit);
        if (document.body.dataset.sharedKeySettings === 'true') {
            document.getElementById('apiKeyProviderSelect')?.addEventListener('change', () => {
                document.getElementById('apiKeyInput').value = '';
                document.getElementById('apiKeyInput').removeAttribute('aria-invalid');
                clearSharedKeyStatus();
                renderSharedKeySettings(providerSettings);
            });
            document.getElementById('saveSettingsBtn')?.addEventListener('click', saveSharedKey);
        }
        document.querySelectorAll('[data-inspector-close]').forEach(button => button.addEventListener('click', closeInspector));
        document.addEventListener('keydown', event => {
            // Bootstrap owns the focus trap while a confirmation dialog is on top.
            if (document.querySelector('.modal.show') || event.target?.closest?.('.modal')) return;
            if (event.key === 'Escape' && closeControlHelp) {
                event.preventDefault();
                closeControlHelp();
                return;
            }
            const openInspector = inspector && !inspector.hidden;
            const panel = document.getElementById('controlsPanel');
            const openSidebar = narrow.matches && panel?.classList.contains('show');
            if (event.key === 'Escape') {
                if (openInspector) { event.preventDefault(); closeInspector(); }
                else if (openSidebar) { event.preventDefault(); closeSidebar(); }
            }
            if (event.key !== 'Tab' || !narrow.matches) return;
            const active = openInspector ? inspector : (openSidebar ? panel : null);
            if (!active) return;
            const nodes = focusable(active);
            const first = nodes[0], last = nodes[nodes.length - 1];
            if (!first) return;
            if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
            else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        });
        narrow.addEventListener('change', () => { syncSidebar(); syncInspector(); });
    });

    window.IdeaWorkbench = { finite, providerName, modelProvider, preferredModel, selectedProvider, setProviderSettings, providerSettingsSnapshot: () => providerSettings, loadProviderSettings, updateProviderReadiness, configureModelPicker, selectModel, openProviderSettings, renderSharedKeySettings, editProviderKey, cancelProviderKeyEdit, money, billingCoverageText, renderBilling, showCosts, historySamplingNotice, renderHistorySamplingNote, metricEvidence, renderMetrics, renderPrimaryArtifact, openInspector, closeInspector, closeSidebar, selectSidebarView, renderChartTable };
})();
