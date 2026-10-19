const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appPath = process.env.SAFOD_APP_PATH || path.join(__dirname, "../templates/app.js");
const source = fs.readFileSync(appPath, "utf8");
const event = {
  event_id: "nc75379261",
  latitude: 35.9835,
  longitude: -120.54166,
  origin_time_utc: "2026-06-18T21:04:42.290000Z",
  magnitude: 1.6,
  depth_km: 3.43,
  event_url: "https://earthquake.usgs.gov/earthquakes/eventpage/nc75379261/",
  plot_available: true,
  plots: [{ url: "plots/75379261.webp" }],
};
const otherEvent = { ...event, event_id: "nc75334592", latitude: 35.95, longitude: -120.52, plot_available: false, plots: [] };
const nextTurn = () => new Promise((resolve) => setImmediate(resolve));

function element() {
  const listeners = new Map();
  const classes = new Set(["hidden"]);
  return {
    value: "",
    textContent: "",
    open: false,
    checked: false,
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
    },
    addEventListener(name, callback) { listeners.set(name, callback); },
    dispatch(name, value = {}) { listeners.get(name)?.(value); },
    appendChild() {},
    setAttribute(name, value) { this[name] = String(value); },
    removeAttribute(name) { delete this[name]; },
    showModal() { this.open = true; },
    close() { this.open = false; },
  };
}

function startApp(hash = "", { delayLayers = false, delayCatalog = false, missingFocal = false } = {}) {
  const elements = new Map();
  const windowListeners = new Map();
  const errors = [];
  const actions = [];
  let view = { zoom: 10 };
  let popupCoordinates;
  const layers = new Set();
  const pendingLayers = [];
  let releaseCatalog;
  const catalogReady = new Promise((resolve) => { releaseCatalog = resolve; });
  const location = { hash, pathname: "/safod-event-explorer/", search: "" };
  const getElement = (id) => {
    if (!elements.has(id)) {
      const node = element();
      if (id === "eventDialog") {
        node.showModal = function () {
          actions.push("modal");
          this.open = true;
        };
      }
      elements.set(id, node);
    }
    return elements.get(id);
  };
  const chain = () => {
    const ownedLayers = new Set();
    return ({
    addTo() { return this; },
    setView(coordinates, zoom, options) {
      view = { coordinates: Array.from(coordinates), zoom, animate: options?.animate };
      actions.push("view");
      return this;
    },
    getZoom() { return view.zoom; },
    openPopup() {
      assert.ok(layers.has(this), "A popup requires a visible marker");
      popupCoordinates = Array.from(this.coordinates);
      actions.push("popup");
      return this;
    },
    bringToFront() { return this; },
    on() { return this; },
    bindTooltip() { this.tooltip = {}; return this; },
    bindPopup() { this.popup = {}; return this; },
    getTooltip() { return this.tooltip; },
    getPopup() { return this.popup; },
    hasLayer: (layer) => ownedLayers.has(layer),
    addLayer(layer) { ownedLayers.add(layer); layers.add(layer); return this; },
    removeLayer(layer) { ownedLayers.delete(layer); layers.delete(layer); return this; },
    eachLayer: (callback) => ownedLayers.forEach(callback),
    clearLayers() {
      ownedLayers.forEach((layer) => layers.delete(layer));
      ownedLayers.clear();
      return this;
    },
    });
  };
  const L = {
    map: chain, tileLayer: chain, layerGroup: chain, geoJSON: chain,
    circleMarker: (coordinates) => Object.assign(chain(), { coordinates }),
  };
  const data = {
    "data/events.json": [event, otherEvent],
    "data/site_status.json": { recorded_events: 2, plots_available: 1 },
    "data/site_config.json": { map_center: [35.9, -120.5], map_default_zoom: 10 },
    "data/focal_mechanisms.json": {},
    "data/safod_geometry.geojson": { type: "FeatureCollection", features: [] },
    "data/san_andreas_fault.geojson": {
      type: "FeatureCollection",
      features: [{ properties: { fault_name: "San Andreas fault zone" } }],
    },
  };

  vm.runInNewContext(source, {
    window: {
      L,
      SAFOD_BUILD_ID: "test",
      addEventListener: (name, callback) => windowListeners.set(name, callback),
    },
    document: { getElementById: getElement, createElement: element },
    location,
    history: {
      replaceState(_state, _title, url) {
        location.hash = new URL(url, "https://example.org/").hash;
      },
    },
    URLSearchParams,
    L,
    console: { error: (error) => errors.push(error) },
    fetch: async (url) => {
      const name = url.split("?")[0];
      assert.ok(name in data, `Unexpected request: ${url}`);
      if (missingFocal && name === "data/focal_mechanisms.json") {
        return { ok: false, status: 404, statusText: "Not Found" };
      }
      if (delayCatalog && name === "data/events.json") await catalogReady;
      if (delayLayers && name.endsWith(".geojson")) {
        await new Promise((resolve) => pendingLayers.push(resolve));
      }
      return { ok: true, json: async () => data[name] };
    },
  }, { filename: appPath });

  return {
    get: getElement,
    location,
    errors,
    actions,
    get view() { return view; },
    get popupCoordinates() { return popupCoordinates; },
    setZoom(zoom) { view.zoom = zoom; },
    releaseCatalog,
    releaseLayers: () => pendingLayers.forEach((resolve) => resolve()),
    navigate(hash) {
      location.hash = hash;
      windowListeners.get("hashchange")?.();
    },
  };
}

function assertEventOpen(app, expected = event) {
  assert.deepEqual(app.errors, []);
  assert.equal(app.get("eventDialog").open, true);
  assert.ok(app.get("dialogTitle").textContent.endsWith(expected.event_id));
  assert.equal(app.location.hash, `#event=${expected.event_id}`);
  const coordinates = [expected.latitude, expected.longitude];
  assert.deepEqual(app.view.coordinates, coordinates);
  assert.ok(app.view.zoom >= 13);
  assert.equal(app.view.animate, false);
  assert.deepEqual(app.popupCoordinates, coordinates);
  if (expected.plot_available) assert.equal(app.get("dialogPlot").src, `${expected.plots[0].url}?v=test`);
}

for (const id of ["nc75379261", "75379261", "NC75379261"]) {
  test(`opens the requested event on initial load: ${id}`, async () => {
    const app = startApp(`#event=${id}`);
    await nextTurn();
    assertEventOpen(app);
  });
}

test("opens an event when the optional focal-mechanism catalog is absent", async () => {
  const app = startApp("#event=nc75379261", { missingFocal: true });
  await nextTurn();
  assertEventOpen(app);
});

test("opens the event before optional map layers finish loading", async () => {
  const app = startApp("#event=nc75379261", { delayLayers: true });
  await nextTurn();
  try {
    assertEventOpen(app);
    app.get("closeDialog").dispatch("click");
  } finally {
    app.releaseLayers();
  }
  await nextTurn();
  assert.equal(app.get("visibleCount").textContent, "2 / 2");
  assert.equal(app.get("eventDialog").open, false);
});

test("opens and switches events when the hash changes in the same tab", async () => {
  const app = startApp();
  await nextTurn();
  assert.equal(app.get("eventDialog").open, false);
  app.navigate("#event=nc75379261");
  assertEventOpen(app);
  app.navigate("#event=nc75334592");
  assertEventOpen(app, otherEvent);
  assert.equal(app.get("dialogPlot").src, undefined);
  assert.equal(app.get("plotUnavailable").classList.contains("hidden"), false);
});

test("uses the latest hash when the catalog finishes loading", async () => {
  const app = startApp("#event=nc75334592", { delayCatalog: true });
  app.navigate("#event=nc75379261");
  app.releaseCatalog();
  await nextTurn();
  assertEventOpen(app);
});

test("close button and Escape clear the event link and allow reopening", async () => {
  const app = startApp("#event=nc75379261");
  await nextTurn();
  assertEventOpen(app);
  app.get("closeDialog").dispatch("click");
  assert.equal(app.location.hash, "");
  assert.equal(app.get("eventDialog").open, false);
  app.navigate("#event=nc75379261");
  assertEventOpen(app);
  app.get("eventDialog").dispatch("cancel", { preventDefault() {} });
  assert.equal(app.location.hash, "");
  assert.equal(app.get("eventDialog").open, false);
});

test("removing the event hash closes the dialog", async () => {
  const app = startApp("#event=nc75379261");
  await nextTurn();
  app.navigate("");
  assert.equal(app.get("eventDialog").open, false);
  assert.equal(app.get("dialogPlot").src, undefined);
});

test("an invalid encoded event ID does not stop the explorer", async () => {
  const app = startApp("#event=%E0%A4%A");
  await nextTurn();
  assert.deepEqual(app.errors, []);
  assert.equal(app.get("eventDialog").open, false);
  assert.equal(app.get("visibleCount").textContent, "2 / 2");
});


test("centers the map and opens the marker popup before the DAS modal", async () => {
  const app = startApp("#event=nc75379261");
  await nextTurn();
  assertEventOpen(app);
  assert.equal(app.view.zoom, 13);
  assert.deepEqual(app.actions, ["view", "view", "popup", "modal"]);
});

test("keeps a closer zoom when navigating to an event", async () => {
  const app = startApp();
  await nextTurn();
  app.setZoom(16);
  app.navigate("#event=nc75379261");
  assertEventOpen(app);
  assert.equal(app.view.zoom, 16);
});

test("reveals an event hidden by filters before opening its popup", async () => {
  const app = startApp();
  await nextTurn();
  app.get("searchInput").value = otherEvent.event_id;
  app.get("searchInput").dispatch("input");
  assert.equal(app.get("visibleCount").textContent, "1 / 2");
  app.navigate("#event=nc75379261");
  assertEventOpen(app);
  assert.equal(app.get("searchInput").value, "");
  assert.equal(app.get("visibleCount").textContent, "2 / 2");
});
