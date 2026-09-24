// IndexedDB cache for lightweight face crops and WebP thumbnails.
// Avoids repeated network transfers or canvas extractions on reload and re-selection.
// Falls back to an in-memory Map when IndexedDB is unavailable (SSR, test runners, or private mode).

const DB_NAME = 'roop_facebank_cache';
const DB_VERSION = 1;
const STORE_NAME = 'crops';

// In-memory fallback cache
const memoryCache = new Map();

let dbPromise = null;

/**
 * Open or initialize the IndexedDB instance.
 * @returns {Promise<IDBDatabase|null>}
 */
export function getFaceBankDb() {
  if (typeof window === 'undefined' || !window.indexedDB) {
    return Promise.resolve(null);
  }

  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve) => {
    try {
      const request = window.indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };

      request.onsuccess = (event) => {
        resolve(event.target.result);
      };

      request.onerror = () => {
        resolve(null);
      };
    } catch {
      resolve(null);
    }
  });

  return dbPromise;
}

/**
 * Convert a base64 or DataURL string to a Blob.
 * @param {string} dataUrl
 * @returns {Blob|null}
 */
export function dataUrlToBlob(dataUrl) {
  if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:')) {
    return null;
  }
  try {
    const parts = dataUrl.split(',');
    const mimeMatch = parts[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg';
    const binaryStr = atob(parts[1]);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    return new Blob([bytes], { type: mime });
  } catch {
    return null;
  }
}

/**
 * Store a crop (Blob or DataURL) in the cache.
 * @param {string} key Unique identifier for the crop (e.g. cluster ID or hash)
 * @param {Blob|string} data Blob or base64 dataUrl
 * @param {Object} metadata Extra metadata (dimensions, score, etc.)
 * @returns {Promise<boolean>}
 */
export async function cacheCrop(key, data, metadata = {}) {
  if (!key || !data) return false;

  const record = {
    key: String(key),
    data,
    timestamp: Date.now(),
    metadata: metadata || {},
  };

  // Keep in memory fallback as well for speed
  memoryCache.set(String(key), record);

  const db = await getFaceBankDb();
  if (!db) return true;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(record);
      req.onsuccess = () => resolve(true);
      req.onerror = () => resolve(false);
    } catch {
      resolve(false);
    }
  });
}

/**
 * Retrieve a cached crop by key.
 * @param {string} key
 * @returns {Promise<{ key: string, data: Blob|string, metadata: Object, timestamp: number }|null>}
 */
export async function getCachedCrop(key) {
  if (!key) return null;
  const strKey = String(key);

  // Check memory cache first
  if (memoryCache.has(strKey)) {
    return memoryCache.get(strKey);
  }

  const db = await getFaceBankDb();
  if (!db) return null;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(strKey);
      req.onsuccess = () => {
        const result = req.result || null;
        if (result) {
          memoryCache.set(strKey, result);
        }
        resolve(result);
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

/**
 * Remove a specific cached crop.
 * @param {string} key
 * @returns {Promise<boolean>}
 */
export async function deleteCachedCrop(key) {
  if (!key) return false;
  const strKey = String(key);
  memoryCache.delete(strKey);

  const db = await getFaceBankDb();
  if (!db) return true;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(strKey);
      req.onsuccess = () => resolve(true);
      req.onerror = () => resolve(false);
    } catch {
      resolve(false);
    }
  });
}

/**
 * Clear the entire crop cache.
 * @returns {Promise<boolean>}
 */
export async function clearCropCache() {
  memoryCache.clear();
  const db = await getFaceBankDb();
  if (!db) return true;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.clear();
      req.onsuccess = () => resolve(true);
      req.onerror = () => resolve(false);
    } catch {
      resolve(false);
    }
  });
}

/**
 * Get cache count and storage diagnostics.
 * @returns {Promise<{ count: number, isAvailable: boolean }>}
 */
export async function getCacheStats() {
  const db = await getFaceBankDb();
  if (!db) {
    return { count: memoryCache.size, isAvailable: false };
  }

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.count();
      req.onsuccess = () => {
        resolve({ count: req.result, isAvailable: true });
      };
      req.onerror = () => {
        resolve({ count: memoryCache.size, isAvailable: false });
      };
    } catch {
      resolve({ count: memoryCache.size, isAvailable: false });
    }
  });
}
