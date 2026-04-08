from __future__ import annotations

import re


async def parse_chatgpt_link(url: str) -> list[dict]:
    """Parse a ChatGPT share link and extract user/assistant turns.

    Uses Playwright to render the JS-heavy page and extract via
    data-message-author-role attributes.
    """
    if not re.match(r"https?://chatgpt\.com/share/", url):
        raise ValueError("Invalid ChatGPT share URL")

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=30000)

        messages = await page.query_selector_all("[data-message-author-role]")
        turns = []
        for msg in messages:
            role = await msg.get_attribute("data-message-author-role")
            text = await msg.inner_text()
            turns.append({"role": role, "text": text[:2000]})

        await browser.close()

    return turns


def parse_pasted_text(text: str) -> list[dict]:
    """Parse pasted conversation text into turns.

    Supports formats:
    - Human: ... / Assistant: ...
    - User: ... / AI: ...
    - You: ... / ChatGPT: ...
    """
    user_patterns = re.compile(
        r"^(?:Human|User|You|사용자)\s*:\s*", re.MULTILINE | re.IGNORECASE
    )
    asst_patterns = re.compile(
        r"^(?:Assistant|AI|ChatGPT|Claude|GPT|어시스턴트)\s*:\s*",
        re.MULTILINE | re.IGNORECASE,
    )

    # Split by role markers
    combined = re.compile(
        r"^((?:Human|User|You|사용자|Assistant|AI|ChatGPT|Claude|GPT|어시스턴트)\s*:)",
        re.MULTILINE | re.IGNORECASE,
    )

    parts = combined.split(text)
    if len(parts) < 2:
        # No markers found — treat each line as a user turn
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return [{"role": "user", "text": l} for l in lines]

    turns = []
    i = 1
    while i < len(parts) - 1:
        marker = parts[i].strip().rstrip(":")
        content = parts[i + 1].strip()
        if user_patterns.match(parts[i]):
            role = "user"
        elif asst_patterns.match(parts[i]):
            role = "assistant"
        else:
            role = "user"
        if content:
            turns.append({"role": role, "text": content[:2000]})
        i += 2

    return turns
