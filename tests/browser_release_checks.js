/* Free browser recovery checks. All run-launch requests are intercepted. */
async (page) => {
    const base = 'http://127.0.0.1:8000';
    const failures = [];
    const check = (condition, message) => { if (!condition) failures.push(message); };
    const runtimeErrors = [];
    const onError = error => runtimeErrors.push(error.message);
    page.on('pageerror', onError);
    await page.setViewportSize({ width: 1440, height: 1000 });
    try {
        await page.goto(base);
        await page.waitForSelector('.template-library-select');
        await page.locator('.template-library-select').filter({ hasText: 'Drabble' }).click();
        await page.locator('#evolutionNameInput').fill('Recovery audit');
        await page.route('**/api/start-evolution', route => route.fulfill({
            status: 422, contentType: 'application/json',
            body: JSON.stringify({ detail: [{ loc: ['body', 'popSize'], msg: 'Test validation failure' }] }),
        }));
        await page.locator('#startButton').click();
        await page.waitForFunction(() => document.getElementById('workspaceNotice')?.textContent.includes('popSize'));
        check((await page.locator('#workspaceNotice').innerText()).includes('Test validation failure'), 'validation detail is visible');
        check(await page.locator('#evolutionNameInput').inputValue() === 'Recovery audit', 'rejected start preserves name');
        check(await page.locator('#startButton').isEnabled(), 'rejected start permits correction');
        check(await page.locator('#stopButton').isHidden(), 'rejected start does not offer Stop');

        await page.unroute('**/api/start-evolution');
        await page.route('**/api/start-evolution', route => route.abort('failed'));
        await page.locator('#startButton').click();
        await page.waitForFunction(() => document.getElementById('workspaceNotice')?.textContent.includes('could not confirm'));
        check(await page.locator('#evolutionNameInput').inputValue() === 'Recovery audit', 'lost launch response preserves name');
        await page.unroute('**/api/start-evolution');

        let progressRequests = 0;
        await page.route('**/api/progress**', route => {
            progressRequests++;
            if (progressRequests === 1) return route.abort('failed');
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
                evolution_id: 'offline-recovery-audit', evolution_name: 'Recovery audit',
                is_running: false, status: 'error', error: 'Provider unavailable',
                is_resumable: false, history: [], progress: 20,
            }) });
        });
        await page.route('**/api/start-evolution', route => route.fulfill({
            status: 200, contentType: 'application/json', body: JSON.stringify({
                evolution_id: 'offline-recovery-audit', evolution_name: 'Recovery audit',
            }),
        }));
        await page.locator('#startButton').click();
        await page.waitForFunction(() => document.getElementById('workspaceNotice')?.textContent.includes('Connection lost'));
        await page.waitForFunction(() => document.getElementById('evolution-status-text')?.textContent === 'Failed', null, { timeout: 15000 });
        check(!(await page.locator('#activity-log-items').innerText()).includes('Completed'), 'failed run does not announce success');
        check((await page.locator('#workspaceNotice').innerText()).includes('Provider unavailable'), 'failure remains visible');
        check(await page.locator('#startButton').isEnabled(), 'failed run releases launch control');
        check(progressRequests === 2, 'one reconnect loop, ending after terminal failure');
        await page.screenshot({ path: 'output/playwright/release-recovery-failure.png' });
        await page.unroute('**/api/start-evolution');
        await page.unroute('**/api/progress**');

        await page.goto(base);
        await page.waitForSelector('.template-library-select');
        await page.route('**/api/settings/status', route => route.fulfill({
            status: 503, contentType: 'application/json', body: JSON.stringify({ message: 'Temporarily unavailable' }),
        }));
        await page.locator('button[onclick="showSidebarView(\'settings\')"]').click();
        await page.waitForFunction(() => document.getElementById('settingsStatus')?.textContent.includes('Could not refresh'));
        check(await page.locator('#settingsStatus').isVisible(), 'key status failure is visible and retryable');
        await page.unroute('**/api/settings/status');
        await page.locator('#view-settings [onclick="showSidebarView(\'main\')"]').click();
        await page.locator('button[onclick="showSidebarView(\'settings\')"]').click();
        await page.waitForFunction(() => !document.getElementById('view-settings').hasAttribute('aria-busy'));
        check(await page.locator('#settingsStatus').isHidden(), 'successful retry clears stale error');
        check(runtimeErrors.length === 0, `runtime errors: ${runtimeErrors.join('; ')}`);
        if (failures.length) throw new Error(failures.join('\n'));
        return { result: 'passed', scenarios: 8, progressRequests, runtimeErrors, paidRequests: 0 };
    } finally {
        await page.unroute('**/api/start-evolution');
        await page.unroute('**/api/progress**');
        await page.unroute('**/api/settings/status');
        page.off('pageerror', onError);
    }
}
