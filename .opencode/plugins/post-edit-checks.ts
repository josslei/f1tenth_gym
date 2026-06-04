import type { Plugin } from "@opencode-ai/plugin"

const FILE_EDIT_TOOLS = new Set(["edit", "write", "apply_patch"])

function extractPatchFiles(patchText: string): string[] {
  const files = new Set<string>()
  for (const line of patchText.split("\n")) {
    const match = line.match(/^\*\*\* (?:Add|Update|Move to) File: (.+)$/)
    if (match) files.add(match[1].trim())
  }
  return [...files]
}

export const PostEditChecks: Plugin = async ({ $, directory }) => {
  const preCommitConfig = `${directory}/.pre-commit-config.yaml`

  return {
    "tool.execute.after": async (input, output) => {
      if (!FILE_EDIT_TOOLS.has(input.tool)) return

      let files: string[] = []

      if (input.tool === "apply_patch" && input.args?.patchText) {
        files = extractPatchFiles(input.args.patchText)
      } else if (input.args?.filePath) {
        files = [input.args.filePath]
      }

      if (files.length === 0) return

      const result =
        await $`pre-commit run --files ${files}`
          .cwd(directory)
          .env({ ...process.env })
          .nothrow()
          .quiet()

      if (result.exitCode !== 0) {
        console.warn(
          "[post-edit-checks] pre-commit found issues in edited files"
        )
      }
    },
  }
}
