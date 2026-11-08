import { useEffect, useState } from 'react'

export type Route = 'landing' | 'app'

function read(): Route {
  return window.location.hash.startsWith('#/app') ? 'app' : 'landing'
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(read)
  useEffect(() => {
    const onChange = () => setRoute(read())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
