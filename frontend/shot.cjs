const { chromium } = require("playwright-core")

const OUT = process.env.SHOT_DIR || "/tmp"

async function main() {
  const browser = await chromium.launch({ channel: "chrome" }).catch(() =>
    chromium.launch({ channel: "msedge" })
  )
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  await page.goto("http://localhost:4173/", { waitUntil: "networkidle" })
  await page.waitForSelector("text=Iteration ledger", { timeout: 15000 })

  await page.screenshot({ path: `${OUT}/run-top.png` })
  await page.screenshot({ path: `${OUT}/run-full.png`, fullPage: true })

  const kpis = await page.$$eval(".font-mono", (els) =>
    els.map((e) => e.textContent.trim()).filter((t) => t.length < 30).slice(0, 40)
  )
  console.log(kpis.join(" | "))
  await browser.close()
}

main().catch((e) => {
  console.error("shot failed:", e.message)
  process.exit(1)
})
