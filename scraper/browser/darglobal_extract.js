/*
 * DarGlobal extractor — run in the DevTools console of an open https://darglobal.co.uk tab.
 *
 * Why in-browser?  darglobal.co.uk sits behind Imperva/Incapsula bot protection, so plain
 * HTTP clients get a JS challenge. Running inside a normal browser session (same-origin
 * fetches, one page every 0.8s) reads the public, server-rendered pages exactly like a
 * visitor would — no protection is bypassed. robots.txt allows /, /projects/, /press/, /blog/.
 *
 * Output: downloads darglobal_raw.json  ->  move it to data/raw/
 */
(async () => {
  const DELAY_MS = 800;
  const SKIP = /\/(blog|ar)(\/|$)|thank-you|success|pay-online|internship|landing|privacy|terms|bookYourUnit|campaigns|win-a-trip|broker-contest|careers|become-a|new-home-page|cookie|get-in-touch|Invest-crypto|DarGlobal-Insignia-Card|DarGlobal-exclusive-Off-Plan/i;

  const sitemap = await (await fetch('/sitemap.xml')).text();
  const urls = [...new Set([...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1].trim()))]
    .filter(u => !SKIP.test(u.replace('https://darglobal.co.uk', '')));

  const out = { source: 'darglobal.co.uk', scrapedAt: new Date().toISOString(), projectList: null, errors: [], pages: [] };
  for (const [i, u] of urls.entries()) {
    try {
      const r = await fetch(u.replace('https://darglobal.co.uk', ''));
      const html = await r.text();
      const doc = new DOMParser().parseFromString(html, 'text/html');

      // Structured CMS data embedded by Next.js
      let project = null;
      const nd = html.match(/<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/);
      if (nd) {
        const pp = JSON.parse(nd[1]).props.pageProps;
        if (!out.projectList && pp.globalData) out.projectList = pp.globalData.attributes.ProjectList;
        const a = pp.projectDetailsData && pp.projectDetailsData.attributes;
        if (a) project = { title: a.title, slug: a.slug, projectCode: a.projectCode, details: a.ProjectDetails, faq: a.faqContent };
      }

      // Visible text (minus navigation, forms, footer)
      doc.querySelectorAll('script,style,noscript,form,footer,header,nav,svg,iframe,select').forEach(e => e.remove());
      doc.querySelectorAll('p,h1,h2,h3,h4,h5,h6,li,div,br,section,tr,dt,dd,span').forEach(e => e.append('\n'));
      let text = (doc.body ? doc.body.textContent : '').replace(/[ \t ]+/g, ' ').replace(/\n\s*/g, '\n').replace(/\n{2,}/g, '\n').trim();
      for (const cut of ['REGISTER YOUR INTEREST', 'Register Your Interest', 'Discover Luxury Properties', 'We value our privacy']) {
        const k = text.indexOf(cut); if (k > 300) text = text.slice(0, k);
      }
      out.pages.push({ url: u, status: r.status, title: doc.title, metaDesc: doc.querySelector('meta[name="description"]')?.content || '', text, project });
    } catch (e) { out.errors.push([u, String(e)]); }
    console.log(`[darglobal] ${i + 1}/${urls.length}`);
    await new Promise(res => setTimeout(res, DELAY_MS));
  }

  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], { type: 'application/json' }));
  a.download = 'darglobal_raw.json'; a.click();
})();
