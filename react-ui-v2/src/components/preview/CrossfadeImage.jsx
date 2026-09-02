import React, { useEffect, useState } from 'react';

/**
 * Crossfades between image sources using TWO persistent <img> elements.
 * Eliminates frame flashes, white flickers, and DOM remount stutter during scrubbing.
 */
export function CrossfadeImage({
  src,
  alt = '',
  className = '',
  style = {},
  fadeMs = 150,
  onLoad,
}) {
  const [layers, setLayers] = useState({ a: src, b: src, front: 'a' });

  useEffect(() => {
    setLayers((s) => {
      if (src === s[s.front]) return s;
      const back = s.front === 'a' ? 'b' : 'a';
      // If the back layer already has the image, promote it immediately
      if (src === s[back]) return { ...s, front: back };
      return { ...s, [back]: src };
    });
  }, [src]);

  const handleLayerLoad = (which, event) => {
    if (onLoad) onLoad(event);
    setLayers((s) => {
      if (s.front === which || s[which] !== src) return s;
      return { ...s, front: which };
    });
  };

  const renderLayer = (which) => (
    <img
      key={which}
      src={layers[which] || ''}
      alt={alt}
      decoding="async"
      aria-hidden={layers.front !== which}
      draggable={false}
      onLoad={(e) => handleLayerLoad(which, e)}
      className={`absolute inset-0 w-full h-full object-contain pointer-events-none select-none ${className}`}
      style={{
        ...style,
        opacity: layers.front === which ? 1 : 0,
        transition: `opacity ${fadeMs}ms cubic-bezier(0.16, 1, 0.3, 1)`,
      }}
    />
  );

  return (
    <div className="relative w-full h-full">
      {renderLayer('a')}
      {renderLayer('b')}
    </div>
  );
}
