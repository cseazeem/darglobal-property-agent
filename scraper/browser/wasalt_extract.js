/*
 * Wasalt extractor — run in the DevTools console of an open https://wasalt.sa tab.
 *
 * wasalt.sa is behind Cloudflare, so this runs inside a normal browser session: same-origin
 * fetches of public, server-rendered search pages (one every 1.2s). robots.txt disallows
 * /search, so only the public category pages (/en/properties-for-sale-in-<city>) are used.
 * Personal data (agent / owner names, phone numbers, licence numbers) is NOT collected.
 *
 * Output: downloads wasalt_raw.json  ->  move it to data/raw/
 */
(async () => {
  const DELAY_MS = 1200;
  const CITIES = { riyadh: [3, 2], jeddah: [3, 2], makkah: [2, 2], madinah: [2, 2], dammam: [2, 1], 'al-khobar': [2, 1], 'khamis-mushayt': [1, 1] };
  const plan = [];
  for (const [c, [sale, rent]] of Object.entries(CITIES)) {
    for (let i = 1; i <= sale; i++) plan.push(`/en/properties-for-sale-in-${c}` + (i > 1 ? `?page=${i}` : ''));
    for (let i = 1; i <= rent; i++) plan.push(`/en/properties-for-rent-in-${c}` + (i > 1 ? `?page=${i}` : ''));
  }
  ['villas-for-sale-in-riyadh', 'apartments-for-rent-in-riyadh', 'villas-for-sale-in-jeddah',
   'apartments-for-rent-in-jeddah', 'commercial-lands-for-sale-in-riyadh'].forEach(t => plan.push('/en/' + t));

  const html2txt = s => {
    if (!s) return '';
    const d = new DOMParser().parseFromString(s, 'text/html');
    d.querySelectorAll('h1,h2,h3,h4,p,li,br,tr').forEach(e => e.append('\n'));
    return d.body.textContent.replace(/[ \t ]+/g, ' ').replace(/\n\s*/g, '\n').trim();
  };

  const out = { source: 'wasalt.sa', scrapedAt: new Date().toISOString(), cityPages: [], listings: {} };
  for (const [i, p] of plan.entries()) {
    try {
      const html = await (await fetch(p)).text();
      const nd = html.match(/<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/);
      if (!nd) continue;
      const pp = JSON.parse(nd[1]).props.pageProps;
      if (!p.includes('page=')) {
        out.cityPages.push({
          url: 'https://wasalt.sa' + p, total: (pp.searchResult || {}).count,
          about: html2txt(typeof pp.para === 'string' ? pp.para : '').slice(0, 6000),
          faq: ((pp.FAQSchema || {}).mainEntity || []).map(q => ({ q: q.name, a: (q.acceptedAnswer || {}).text })),
        });
      }
      for (const x of (pp.searchResult || {}).properties || []) {
        const info = x.propertyInfo || {};
        const attrs = {}; for (const a of x.attributes || []) attrs[a.key] = a.value + (a.unit ? ' ' + a.unit : '');
        out.listings[x.id] = {
          id: x.id, url: `https://wasalt.sa/en/property/${info.propertyFor}/${info.slug}`, title: info.title,
          propertyFor: info.propertyFor, mainType: info.propertyMainType, subType: info.propertySubType,
          city: info.city, territory: info.territory, district: info.district || info.zone, address: info.address,
          price: info.propertyFor === 'rent' ? info.expectedRent : info.salePrice, rentFreq: info.rentFreq || null,
          currency: info.currencyType || 'SAR', attributes: attrs, furnishing: info.furnishingType || null,
          projectName: info.projectName || null, companyRole: (x.propertyOwner || {}).enUserRole || null,
          publishedAt: x.publishedAt, ageOfListing: x.ageOfListing,
          lat: (x.location || {}).lat, lon: (x.location || {}).lon, source: p,
        };
      }
    } catch (e) { console.warn(p, e); }
    console.log(`[wasalt] ${i + 1}/${plan.length}`);
    await new Promise(res => setTimeout(res, DELAY_MS));
  }

  out.listings = Object.values(out.listings);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], { type: 'application/json' }));
  a.download = 'wasalt_raw.json'; a.click();
})();
