/** Blockly import shim: ESM namespace in the browser, CommonJS default in Node (tests). */
import * as BlocklyNS from 'blockly/core';

type BlocklyType = typeof BlocklyNS;
const ns = BlocklyNS as unknown as BlocklyType & { default?: BlocklyType };
const Blockly: BlocklyType = (ns.Workspace ? ns : ns.default) as BlocklyType;
export default Blockly;
