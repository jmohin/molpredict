import { Editor } from 'ketcher-react'
import { StandaloneStructServiceProvider } from 'ketcher-standalone'
import type { Ketcher } from 'ketcher-core'
import 'ketcher-react/dist/index.css'

const structServiceProvider = new StandaloneStructServiceProvider()

type Props = {
  onReady: (ketcher: Ketcher) => void
  onChange: () => void
}

export function MoleculeEditor({ onReady, onChange }: Props) {
  return (
    <Editor
      staticResourcesUrl=""
      structServiceProvider={structServiceProvider}
      errorHandler={(message) => console.error('Ketcher:', message)}
      disableMacromoleculesEditor
      onInit={(ketcher) => {
        ;(window as unknown as { ketcher: Ketcher }).ketcher = ketcher
        ketcher.editor.subscribe('change', onChange)
        onReady(ketcher)
      }}
    />
  )
}
