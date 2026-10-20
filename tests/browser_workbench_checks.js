/* Run against tests/browser_fixture_app.py with the Playwright browser tool. */
async (page) => {
    const base = 'http://127.0.0.1:8124';
    const failures = [];
    const check = (condition, description) => { if (!condition) failures.push(description); };
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(base);
    await page.waitForSelector('.template-library-select');
    check(await page.locator('.template-library-select').count() === 3, 'template catalog loads');
    check(await page.locator('#welcomeModal').isHidden(), 'first use is unblocked');
    check(await page.locator('.sidebar-view:not(.active)').evaluateAll(nodes => nodes.every(node => node.hidden && node.inert)), 'inactive sidebar views are inert');
    await page.locator('.recent-experiment').click();
    await page.waitForSelector('.evolution-tree-node');
    check(await page.locator('#existingTemplates').isHidden(), 'history spotlights the population');
    await page.locator('.evolution-tree-node').first().click();
    check(await page.locator('#ideaModal').getAttribute('role') === 'complementary', 'desktop inspector is persistent');
    check(await page.locator('.modal-backdrop').count() === 0, 'desktop population stays interactive');
    check((await page.locator('#candidateMetricEvidence').innerText()).includes('weight 1'), 'evaluator weights are visible');
    await page.locator('#evolution-view-tab-ranked').click();
    await page.locator('[data-rank-sort="eval:clarity.testability"]').click();
    check((await page.locator('.evolution-rank-row').first().innerText()).includes('The prime garden, with a control'), 'custom evaluator drives ranking');
    check(await page.locator('#fitnessChartTable tbody tr').count() === 2, 'chart has exact table equivalent');
    await page.keyboard.press('Escape');
    check(await page.locator('#ideaModal').isHidden(), 'Escape closes inspector');

    for (const width of [390, 600, 768]) {
        await page.setViewportSize({ width, height: 900 });
        await page.goto(base);
        await page.waitForSelector('.template-library-select');
        check(await page.locator('#sidebarToggle').isVisible(), `mobile setup toggle reachable at ${width}`);
        await page.locator('#sidebarToggle').click();
        check(await page.locator('#sidebarToggle').getAttribute('aria-expanded') === 'true', `setup opens at ${width}`);
        await page.locator('.recent-experiment').click();
        await page.waitForSelector('.evolution-tree-node');
        await page.locator('.evolution-tree-node').first().click();
        check(await page.locator('#ideaModal').getAttribute('aria-modal') === 'true', `mobile dialog semantics at ${width}`);
        check(await page.locator('.layout-container').evaluate(node => node.inert), `background inert at ${width}`);
        check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `no horizontal page overflow at ${width}`);
        await page.keyboard.press('Tab');
        check(await page.locator('#ideaModal').evaluate(node => node.contains(document.activeElement)), `focus stays in inspector at ${width}`);
        await page.keyboard.press('Escape');
        check(await page.locator('#ideaModal').isHidden(), `Escape closes at ${width}`);
        check(await page.locator('.layout-container').evaluate(node => !node.inert), `background restored at ${width}`);
        check(await page.locator('.evolution-tree-node').first().evaluate(node => node === document.activeElement), `focus returns to selected candidate at ${width}`);
    }

    for (const width of [900, 1024]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.goto(base);
        await page.waitForSelector('.recent-experiment');
        await page.locator('.recent-experiment').click();
        await page.waitForSelector('.evolution-tree-node');
        await page.locator('.evolution-tree-node').first().click();
        check(await page.locator('#ideaModal').isVisible(), `tablet inspector opens at ${width}`);
        check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `no tablet page overflow at ${width}`);
        const workspace = await page.locator('#mainContent').boundingBox();
        check(workspace.width >= 500, `tablet workspace keeps a usable width at ${width}`);
        await page.keyboard.press('Escape');
        check(await page.locator('#ideaModal').isHidden(), `Escape closes tablet inspector at ${width}`);
        check(await page.locator('.evolution-tree-node').first().evaluate(node => node === document.activeElement), `tablet focus returns at ${width}`);
    }

    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(`${base}/rate`);
    await page.waitForSelector('#evolutionSelect option[value="fixture-run"]', { state: 'attached' });
    await page.locator('#evolutionSelect').selectOption('fixture-run');
    await page.waitForFunction(() => document.getElementById('titleA')?.textContent);
    check(await page.locator('#manualVoteBar').isVisible(), 'voting sits beside candidates');
    await page.getByRole('button', { name: 'Show Ranking', exact: true }).click();
    await page.waitForSelector('.idea-details-trigger');
    await page.locator('.idea-details-trigger').first().click();
    check(await page.locator('#ideaModal').isVisible(), 'rater uses the persistent inspector');
    check(await page.locator('#raterChartTable tbody tr').count() > 0, 'rater chart has table equivalent');

    for (const [path, providerId, modelId, width] of [
        ['/', 'providerSelect', 'modelType', 390],
        ['/', 'providerSelect', 'modelType', 1440],
        ['/rate', 'ratingProviderSelect', 'modelSelect', 390],
        ['/rate', 'ratingProviderSelect', 'modelSelect', 1440],
    ]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.goto(`${base}${path}`);
        await page.waitForSelector(`#${providerId}`, { state: 'attached' });
        if (width === 390) await page.locator('#sidebarToggle').click();
        if (path === '/rate') await page.locator('#autoModeBtn').click();
        const provider = page.locator(`#${providerId}`);
        const model = page.locator(`#${modelId}`);
        await provider.selectOption('openai');
        check(await model.locator('option').evaluateAll(options => options.every(option => option.value.startsWith('gpt-'))), `OpenAI models filtered on ${path} at ${width}`);
        await model.selectOption('gpt-6-astra');
        await provider.selectOption('google_generative_ai');
        check(await model.locator('option').evaluateAll(options => options.every(option => option.value.startsWith('gemini-'))), `Gemini models filtered on ${path} at ${width}`);
        await provider.selectOption('openai');
        check(await model.inputValue() === 'gpt-6-astra', `last OpenAI model remembered on ${path} at ${width}`);
        await page.waitForFunction(() => document.getElementById('selectedProviderStatus')?.textContent.includes('ready'));
        check((await page.locator('#selectedProviderStatus').innerText()).includes('OpenAI'), `selected provider readiness on ${path} at ${width}`);
        await page.getByRole('button', { name: 'Cost & usage', exact: true }).click();
        await page.waitForSelector('.billing-total');
        check((await page.locator('.billing-total').innerText()).includes('0.0111'), `recorded costs summed on ${path} at ${width}`);
        await page.locator('.billing-item').first().locator('summary').click();
        check(await page.locator('.billing-item').first().getAttribute('open') !== null, `billing details expand on ${path} at ${width}`);
        check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `billing has no page overflow on ${path} at ${width}`);
        await page.locator('#view-costs button').first().click();
        check(await page.locator('#view-main').isVisible(), `billing returns to controls on ${path} at ${width}`);
        if (width === 390) {
            await page.locator('#sidebarToggle').click();
            check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `workspace restored after billing on ${path}`);
        }
    }
    check(errors.length === 0, `browser has no runtime errors: ${errors.join('; ')}`);
    if (failures.length) throw new Error(failures.join('\n'));
    return { result: 'passed', viewports: [1440, 390, 600, 768, 900, 1024], runtimeErrors: errors };
}
