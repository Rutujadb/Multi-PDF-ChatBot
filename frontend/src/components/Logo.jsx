import { Link } from 'react-router-dom'

export default function Logo({ inverted = false, compact = false }) {
  const boxBg = inverted ? 'bg-paper' : 'bg-ink'
  const centerBg = inverted ? 'bg-ink' : 'bg-paper'
  const size = compact ? 'w-9 h-9' : 'w-10 h-10'
  const inner = compact ? 'w-4 h-4' : 'w-5 h-5'
  const center = compact ? 'w-2.5 h-2.5' : 'w-3 h-3'

  return (
    <Link to="/" className="flex items-center gap-3 group">
      <span className={`relative inline-flex items-center justify-center ${size} ${boxBg} rounded-md`}>
        <span className={`absolute ${inner} bg-brand-500 rounded-sm -translate-x-1 -translate-y-1`} />
        <span className={`absolute ${inner} bg-amber2-500 rounded-sm translate-x-1 translate-y-1`} />
        <span className={`relative ${center} ${centerBg} rounded-[2px]`} />
      </span>
      <span className={`font-extrabold ${compact ? 'text-lg hidden sm:inline' : 'text-xl'} h-tight`}>
        Multi-PDF<span className="text-brand-500">·</span>ChatBot
      </span>
    </Link>
  )
}
