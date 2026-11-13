import { useEffect } from 'react'
import Editor from './Editor'
import { EntryRipple } from './components/EntryRipple'
import { useHashRoute } from './hooks/useHashRoute'
import Landing from './pages/Landing'

export default function App() {
  const route = useHashRoute()
  useEffect(() => {
    if (route === 'app') window.scrollTo(0, 0)
  }, [route])
  return route === 'app' ? (
    <>
      <Editor />
      <EntryRipple />
    </>
  ) : (
    <Landing />
  )
}
