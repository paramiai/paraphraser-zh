
        
from tokenizers import Tokenizer, AddedToken, pre_tokenizers, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFKC, Lowercase, Sequence

from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import BertProcessing
from typing import Optional, List, Union

from tokenizers.models import BPE
from tokenizers.implementations.base_tokenizer import BaseTokenizer

SPECIAL_CHARS = (
    
    # Emojis
    
    '©®‼⁉⃣™ℹ↔↕↖↗↘↙↩↪⌚⌛⌨⏏⏩⏪⏫⏬⏭⏮⏯⏰⏱⏲⏳⏸⏹⏺Ⓜ▪▫▶◀◻◼◽◾☀☁☂☃☄☎☑☔☕☘☝☠☢☣☦☪☮☯☸☹☺♀♂♈♉♊♋♌♍♎♏♐♑♒♓♟♠♣♥♦♨♻♾♿⚒⚓⚔⚕⚖⚗⚙⚛⚜⚠⚡⚧⚪⚫⚰⚱⚽⚾⛄⛅⛈⛎⛏⛑⛓⛔⛩⛪⛰⛱⛲⛳⛴⛵⛷⛸⛹⛺⛽✂✅✈✉✊✋✌✍✏✒✔✖✝✡✨✳✴❄❇❌❎❓❔❕❗❣❤➕➖➗➡➰➿⤴⤵⬅⬆⬇⬛⬜⭐⭕〰〽㊗㊙️🀄🃏🅰🅱🅾🅿🆎🆑🆒🆓🆔🆕🆖🆗🆘🆙🆚🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿🈁🈂🈚🈯🈲🈳🈴🈵🈶🈷🈸🈹🈺🉐🉑🌀🌁🌂🌃🌄🌅🌆🌇🌈🌉🌊🌋🌌🌍🌎🌏🌐🌑🌒🌓🌔🌕🌖🌗🌘🌙🌚🌛🌜🌝🌞🌟🌠🌡🌤🌥🌦🌧🌨🌩🌪🌫🌬🌭🌮🌯🌰🌱🌲🌳🌴🌵🌶🌷🌸🌹🌺🌻🌼🌽🌾🌿🍀🍁🍂🍃🍄🍅🍆🍇🍈🍉🍊🍋🍌🍍🍎🍏🍐🍑🍒🍓🍔🍕🍖🍗🍘🍙🍚🍛🍜🍝🍞🍟🍠🍡🍢🍣🍤🍥🍦🍧🍨🍩🍪🍫🍬🍭🍮🍯🍰🍱🍲🍳🍴🍵🍶🍷🍸🍹🍺🍻🍼🍽🍾🍿🎀🎁🎂🎃🎄🎅🎆🎇🎈🎉🎊🎋🎌🎍🎎🎏🎐🎑🎒🎓🎖🎗🎙🎚🎛🎞🎟🎠🎡🎢🎣🎤🎥🎦🎧🎨🎩🎪🎫🎬🎭🎮🎯🎰🎱🎲🎳🎴🎵🎶🎷🎸🎹🎺🎻🎼🎽🎾🎿🏀🏁🏂🏃🏄🏅🏆🏇🏈🏉🏊🏋🏌🏍🏎🏏🏐🏑🏒🏓🏔🏕🏖🏗🏘🏙🏚🏛🏜🏝🏞🏟🏠🏡🏢🏣🏤🏥🏦🏧🏨🏩🏪🏫🏬🏭🏮🏯🏰🏳🏴🏵🏷🏸🏹🏺🐀🐁🐂🐃🐄🐅🐆🐇🐈🐉🐊🐋🐌🐍🐎🐏🐐🐑🐒🐓🐔🐕🐖🐗🐘🐙🐚🐛🐜🐝🐞🐟🐠🐡🐢🐣🐤🐥🐦🐧🐨🐩🐪🐫🐬🐭🐮🐯🐰🐱🐲🐳🐴🐵🐶🐷🐸🐹🐺🐻🐼🐽🐾🐿👀👁👂👃👄👅👆👇👈👉👊👋👌👍👎👏👐👑👒👓👔👕👖👗👘👙👚👛👜👝👞👟👠👡👢👣👤👥👦👧👨👩👪👫👬👭👮👯👰👱👲👳👴👵👶👷👸👹👺👻👼👽👾👿💀💁💂💃💄💅💆💇💈💉💊💋💌💍💎💏💐💑💒💓💔💕💖💗💘💙💚💛💜💝💞💟💠💡💢💣💤💥💦💧💨💩💪💫💬💭💮💯💰💱💲💳💴💵💶💷💸💹💺💻💼💽💾💿📀📁📂📃📄📅📆📇📈📉📊📋📌📍📎📏📐📑📒📓📔📕📖📗📘📙📚📛📜📝📞📟📠📡📢📣📤📥📦📧📨📩📪📫📬📭📮📯📰📱📲📳📴📵📶📷📸📹📺📻📼📽📿🔀🔁🔂🔃🔄🔅🔆🔇🔈🔉🔊🔋🔌🔍🔎🔏🔐🔑🔒🔓🔔🔕🔖🔗🔘🔙🔚🔛🔜🔝🔞🔟🔠🔡🔢🔣🔤🔥🔦🔧🔨🔩🔪🔫🔬🔭🔮🔯🔰🔱🔲🔳🔴🔵🔶🔷🔸🔹🔺🔻🔼🔽🕉🕊🕋🕌🕍🕎🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛🕜🕝🕞🕟🕠🕡🕢🕣🕤🕥🕦🕧🕯🕰🕳🕴🕵🕶🕷🕸🕹🕺🖇🖊🖋🖌🖍🖐🖕🖖🖤🖥🖨🖱🖲🖼🗂🗃🗄🗑🗒🗓🗜🗝🗞🗡🗣🗨🗯🗳🗺🗻🗼🗽🗾🗿😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿🙀🙁🙂🙃🙄🙅🙆🙇🙈🙉🙊🙋🙌🙍🙎🙏🚀🚁🚂🚃🚄🚅🚆🚇🚈🚉🚊🚋🚌🚍🚎🚏🚐🚑🚒🚓🚔🚕🚖🚗🚘🚙🚚🚛🚜🚝🚞🚟🚠🚡🚢🚣🚤🚥🚦🚧🚨🚩🚪🚫🚬🚭🚮🚯🚰🚱🚲🚳🚴🚵🚶🚷🚸🚹🚺🚻🚼🚽🚾🚿🛀🛁🛂🛃🛄🛅🛋🛌🛍🛎🛏🛐🛑🛒\U0001f6d5\U0001f6d6\U0001f6d7🛠🛡🛢🛣🛤🛥🛩🛫🛬🛰🛳🛴🛵🛶\U0001f6f7\U0001f6f8\U0001f6f9\U0001f6fa\U0001f6fb\U0001f6fc\U0001f7e0\U0001f7e1\U0001f7e2\U0001f7e3\U0001f7e4\U0001f7e5\U0001f7e6\U0001f7e7\U0001f7e8\U0001f7e9\U0001f7ea\U0001f7eb\U0001f90c\U0001f90d\U0001f90e\U0001f90f🤐🤑🤒🤓🤔🤕🤖🤗🤘🤙🤚🤛🤜🤝🤞\U0001f91f🤠🤡🤢🤣🤤🤥🤦🤧\U0001f928\U0001f929\U0001f92a\U0001f92b\U0001f92c\U0001f92d\U0001f92e\U0001f92f🤰\U0001f931\U0001f932🤳🤴🤵🤶🤷🤸🤹🤺🤼🤽🤾\U0001f93f🥀🥁🥂🥃🥄🥅🥇🥈🥉🥊🥋\U0001f94c\U0001f94d\U0001f94e\U0001f94f🥐🥑🥒🥓🥔🥕🥖🥗🥘🥙🥚🥛🥜🥝🥞\U0001f95f\U0001f960\U0001f961\U0001f962\U0001f963\U0001f964\U0001f965\U0001f966\U0001f967\U0001f968\U0001f969\U0001f96a\U0001f96b\U0001f96c\U0001f96d\U0001f96e\U0001f96f\U0001f970\U0001f971\U0001f972\U0001f973\U0001f974\U0001f975\U0001f976\U0001f977\U0001f978\U0001f97a\U0001f97b\U0001f97c\U0001f97d\U0001f97e\U0001f97f🦀🦁🦂🦃🦄🦅🦆🦇🦈🦉🦊🦋🦌🦍🦎🦏🦐🦑\U0001f992\U0001f993\U0001f994\U0001f995\U0001f996\U0001f997\U0001f998\U0001f999\U0001f99a\U0001f99b\U0001f99c\U0001f99d\U0001f99e\U0001f99f\U0001f9a0\U0001f9a1\U0001f9a2\U0001f9a3\U0001f9a4\U0001f9a5\U0001f9a6\U0001f9a7\U0001f9a8\U0001f9a9\U0001f9aa\U0001f9ab\U0001f9ac\U0001f9ad\U0001f9ae\U0001f9af\U0001f9b0\U0001f9b1\U0001f9b2\U0001f9b3\U0001f9b4\U0001f9b5\U0001f9b6\U0001f9b7\U0001f9b8\U0001f9b9\U0001f9ba\U0001f9bb\U0001f9bc\U0001f9bd\U0001f9be\U0001f9bf🧀\U0001f9c1\U0001f9c2\U0001f9c3\U0001f9c4\U0001f9c5\U0001f9c6\U0001f9c7\U0001f9c8\U0001f9c9\U0001f9ca\U0001f9cb\U0001f9cd\U0001f9ce\U0001f9cf\U0001f9d0\U0001f9d1\U0001f9d2\U0001f9d3\U0001f9d4\U0001f9d5\U0001f9d6\U0001f9d7\U0001f9d8\U0001f9d9\U0001f9da\U0001f9db\U0001f9dc\U0001f9dd\U0001f9de\U0001f9df\U0001f9e0\U0001f9e1\U0001f9e2\U0001f9e3\U0001f9e4\U0001f9e5\U0001f9e6\U0001f9e7\U0001f9e8\U0001f9e9\U0001f9ea\U0001f9eb\U0001f9ec\U0001f9ed\U0001f9ee\U0001f9ef\U0001f9f0\U0001f9f1\U0001f9f2\U0001f9f3\U0001f9f4\U0001f9f5\U0001f9f6\U0001f9f7\U0001f9f8\U0001f9f9\U0001f9fa\U0001f9fb\U0001f9fc\U0001f9fd\U0001f9fe\U0001f9ff\U0001fa70\U0001fa71\U0001fa72\U0001fa73\U0001fa74\U0001fa78\U0001fa79\U0001fa7a\U0001fa80\U0001fa81\U0001fa82\U0001fa83\U0001fa84\U0001fa85\U0001fa86\U0001fa90\U0001fa91\U0001fa92\U0001fa93\U0001fa94\U0001fa95\U0001fa96\U0001fa97\U0001fa98\U0001fa99\U0001fa9a\U0001fa9b\U0001fa9c\U0001fa9d\U0001fa9e\U0001fa9f\U0001faa0\U0001faa1\U0001faa2\U0001faa3\U0001faa4\U0001faa5\U0001faa6\U0001faa7\U0001faa8\U0001fab0\U0001fab1\U0001fab2\U0001fab3\U0001fab4\U0001fab5\U0001fab6\U0001fac0\U0001fac1\U0001fac2\U0001fad0\U0001fad1\U0001fad2\U0001fad3\U0001fad4\U0001fad5\U0001fad6\U000e0062\U000e0063\U000e0065\U000e0067\U000e006c\U000e006e\U000e0073\U000e0074\U000e0077' 
    
    # Simple Symbols
    
    #'><'
    ',./?!@#_-=+~`"\';:$%*&^()[]{}'
    
    '、。《》「」『』|\\¶§⌘' 
    
    # Fractions
    
    '⅟½⅓⅕⅙⅛⅔⅖⅚⅜¾⅗⅝⅞⅘¼⅐⅑⅒↉%℅‰‱'
    
    # Technicals
    
    '⌀⌂⌃⌄⌅⌆⌇⌈⌉⌊⌋⌌⌍⌎⌏⌐⌑⌒⌓⌔⌕⌖⌗⌘⌙⌚⌛⌜⌝⌞⌟⌠⌡⌢⌣⌤⌥⌦⌧⌨⌫⌬⌭⌮⌯⌰⌱⌲⌳⌴⌵⌶⌷⌸⌹⌺⌻⌼⌽⌾⌿⍀⍁⍂⍃⍄⍅⍆⍇⍈⍉⍊⍋⍌⍍⍎⍏⍐⍑⍒⍓⍔⍕⍖⍗⍘⍙⍚⍛⍜⍝⍞⍟⍠⍡⍢⍣⍤⍥⍦⍧⍨⍩⍪⍫⍬⍭⍮⍯⍰⍱⍲⍳⍴⍵⍶⍷⍸⍹⍺﹘﹝﹞﹟﹡〶␛␡␚␟␘␠␤␋␌␍␎␏␐␑␒␓␔␕␖␗␙␜␝␞␀␁␂␃␄␅␆␇␈␉␊␢␣⎋'
    
    # Rectangles
    
    '❏❐❑❒▀▁▂▃▄▅▆▇▉▊▋█▌▐▍▎▏▕░▒▓▔▬▢▣▤▥▦▧▨▩▪▫▭▮▯☰☲☱☴☵☶☳☷▰▱◧◨◩◪◫∎■□⊞⊟⊠⊡❘❙❚〓◊◈◇◆⎔⎚☖☗'
    
    # Triangles
    
    '◄▲▼►◀◣◥◤◢▶◂▴▾▸◁△▽▷∆∇⊳⊲⊴⊵◅▻▵▿◃▹◭◮⫷⫸⋖⋗⋪⋫⋬⋭⊿◬≜⑅'
    
    # Lines 
    
    '│┃╽╿╏║╎┇︱┊︳┋┆╵〡〢╹╻╷〣☰☱☲☳☴☵☶☷≡✕═━─╍┅┉┄┈╌╴╶╸╺╼╾﹉﹍﹊﹎︲⑆⑇⑈⑉⑊⑄⑀︴﹏﹌﹋╳╲╱︶︵〵〴〳〆`ᐟ‐⁃⎯〄'
    
    # Corners
    
    '﹄﹃﹂﹁┕┓└┐┖┒┗┑┍┙┏┛┎┚┌┘「」『』˩˥├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╒╕╓╖╔╗╘╛╙╜╚╝╞╡╟╢╠╣╥╨╧╤╦╩╪╫╬〒⊢⊣⊤⊥╭╮╯╰⊦⊧⊨⊩⊪⊫⊬⊭⊮⊯⊺〦〧〨˦˧˨⑁⑂⑃∟'
    
    # Circles
    
    '◉○◌◍◎●◐◑◒◓◔◕◖◗❂☢⊗⊙◘◙◚◛◜◝◞◟◠◡◯〇〶⚫⬤◦∅∘⊕⊖⊘⊚⊛⊜⊝❍⦿'
    
    # Comparisons
    
    '≤≥≦≧≨≩⊰⊱⋛⋚≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣'
    
    # Numerals
    '⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛⓿❶❷❸❹❺❻❼❽❾❿➀➁➂➃➄➅➆➇➈➉⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓪①②③④⑤⑥⑦⑧⑨⑩⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇➊➋➌➍➎➏➐➑➒➓⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴'
    
    'ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻⅼⅽⅾⅿↀↁↂ➀➁➂➃➄➅➆➇➈➉➊➋➌➍➎➏➐➑➒➓⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⓿❶❷❸❹❺❻❼❽❾❿⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉０１２３４５６７８９ⁱₐₑₒₓₔ'
    
    # Currencies
    
    '₳฿￠₡¢₢₵₫€￡£₤₣ƒ₲₭₥₦₱＄$₮₩￦¥￥₴¤₰៛₪₯₠₧₨௹﷼৲৳₹' 
    
    # Mathematic Symbols
    
    '∞⟀⟁⟂⟃⟄⟇⟈⟉⟊⟐⟑⟒⟓⟔⟕⟖⟗⟘⟙⟚⟛⟜⟝⟞⟟⟠⟡⟢⟣⟤⟥⟦⟧⟨⟩⟪⟫⦀⦁⦂⦃⦄⦅⦆⦇⦈⦉⦊⦋⦌⦍⦎⦏⦐⦑⦒⦓⦔⦕⦖⦗⦘⦙⦚⦛⦜⦝⦞⦟⦠⦡⦢⦣⦤⦥⦦⦧⦨⦩⦪⦫⦬⦭⦮⦯⦰⦱⦲⦳⦴⦵⦶⦷⦸⦹⦺⦻⦼⦽⦾⦿⧀⧁⧂⧃⧄⧅⧆⧇⧈⧉⧊⧋⧌⧍⧎⧏⧐⧑⧒⧓⧔⧕⧖⧗⧘⧙⧚⧛⧜⧝⧞⧟⧡⧢⧣⧤⧥⧦⧧⧨⧩⧪⧫⧬⧭⧮⧯⧰⧱⧲⧳⧴⧵⧶⧷⧸⧹⧺⧻⧼⧽⧾⧿∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿⊀⊁⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋⊌⊍⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙⊚⊛⊜⊝⊞⊟⊠⊡⊢⊣⊤⊥⊦⊧⊨⊩⊪⊫⊬⊭⊮⊯⊰⊱⊲⊳⊴⊵⊶⊷⊸⊹⊺⊻⊼⊽⊾⊿⋀⋁⋂⋃⋄⋅⋆⋇⋈⋉⋊⋋⋌⋍⋎⋏⋐⋑⋒⋓⋔⋕⋖⋗⋘⋙⋚⋛⋜⋝⋞⋟⋠⋡⋢⋣⋤⋥⋦⋧⋨⋩⋪⋫⋬⋭⋮⋯⋰⋱⋲⋳⋴⋵⋶⋷⋸⋹⋺⋻⋼⋽⋾⋿✕✖✚÷°'
    
    # Maths
    
    'π∞Σ√∛∜∫∬∭∮∯∰∱∲∳∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙∝∟∠∡∢∣∤∥∦∧∨∩∪∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿⊀⊁⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋⊌⊍⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙⊚⊛⊜⊝⊞⊟⊠⊡⊢⊣⊤⊥⊦⊧⊨⊩⊪⊫⊬⊭⊮⊯⊰⊱⊲⊳⊴⊵⊶⊷⊸⊹⊺⊻⊼⊽⊾⊿⋀⋁⋂⋃⋄⋅⋆⋇⋈⋉⋊⋋⋌⋍⋎⋏⋐⋑⋒⋓⋔⋕⋖⋗⋘⋙⋚⋛⋜⋝⋞⋟⋠⋡⋢⋣⋤⋥⋦⋧⋨⋩⋪⋫⋬⋭⋮⋯⋰⋱⁺⁻⁼⁽⁾ⁿ₊₋₌₍₎✖﹢﹣＋－／＝÷±×',
    
    # Braille Patterns
    '⠁⠂⠄⠈⠐⠠⠃⠅⠆⠘⠨⠰⠉⠒⠤⠑⠡⠢⠊⠌⠔⠇⠸⠎⠱⠣⠜⠪⠕⠋⠙⠓⠚⠍⠩⠥⠬⠖⠲⠦⠴⠏⠹⠧⠼⠫⠝⠮⠵⠺⠗⠞⠳⠛⠭⠶⠟⠻⠷⠾⠯⠽⠿⡀⡄⡆⡇⡏⡛⡜⡟⡶⡷⡼⡾⡿⢀⢉⢠⢣⢤⢧⢰⢸⢹⢻⢿⣀⣁⣄⣆⣇⣉⣒⣕⣘⣙⣛⣠⣤⣥⣦⣧⣩⣬⣭⣰⣴⣵⣶⣷⣸⣹⣻⣼⣾⣿'
    
    # Zhuyin
    
    'ㄍㄎㄫㄐㄑㄬㄉㄊㄋㄅㄆㄇㄈㄪㄗㄘㄙㄓㄔㄕㄏㄒㄌㄖㄧㄨㄩㄚㄛㄝㄟㄞㄠㄡㄢㄤㄣㄥㄦ'
    
    # Gender
    
    '♀♂☹☺☻☿〠ヅツ㋡웃유üÜتシッ㋛웃̟͟ꑇꐦꐠꐡꐕꌇꌈꉕꈋꈌꆛꆜꃼ☠☃〲〴ϡﭢ⍢⍣⍤⍥⍨⍩ὃὕὣѶӪӫ⚣⚤⚥⚦⚧⚨⚢'
    
    # Musical Symbols
    
    '♩♪♫♬♭♮♯°ø؂≠≭'
    
    # Punctuations
    
    '·‑‒–—―‗‘’‚‛“”„‟•‣․‥…‧′″‴‵‶‷❛❜❝❞ʹʺʻʼʽʾʿˀˁ˂˃˄˅ˆˇˈˉˊˋˌˍˎˏːˑ˒˓˔˕˖˗˘˙˚˛˜˝˞ˠˡ～¿﹐﹒﹔﹕！＃＄％＆＊，．：；？＠、。〃〝〞︰'
    
    # Ticks / Cross
    
    '✓✔✗✘☓∨√✇☐☑☒〤〥',

    # Stars

    '★☆≛⋆⍟⍣★☆✡✦✧✪✫✬✯✰✴✵✶✷✸',
    
    # Hearts
    
    '♥♡❤❥❣❦❧დღ۵ლওლ❤️️💙🧡💚💛💜🖤💗💓💔💟💕💖❣️💘💝💞'
    
    # Astrological & Zodiac Sign Symbols
    
    '☮☸♈♉☪♊♋♌♍♎♏♐♑♒♓☤☥☧☨☩☫☬☭☯☽☾✙✚✛✜✝✞✟†⊹‡♁♆❖♅✠✡✢卍卐〷☠☢☣☦'

    # Flowers

    '✽✾✿❀❁❃❊❋✤✣⚘⚜ꕤꕥ☘'
    
    # Arrows
         '☚👈☛👉🖝🖜🖛🖚☜☞🖢👆🖞☝🖣👇🖟☟↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↶↷↸↹↺↻↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏⇕⇖⇗⇘⇙⇚⇛⇜⇝⇞⇟⇠⇡⇢⇣⇤⇥⇦⇧⇨⇩⇪⌅⌆⌤⏎▶☇☈☊☋☌☍➔➘➙➚➛➜➝➞➟➠➡➢➣➤➥➦➧➨➩➪➫➬➭➮➯➱➲➳➴➵➶➷➸➹➺➻➼➽➾⤴⤵↵↓↔←→↑⌦⌫⌧⇰⇫⇬⇭⇳⇮⇯⇱⇲⇴⇵⇷⇸⇹⇺⇑⇓⇽⇾⇿⬳⟿⤉⤈⇻⇼⬴⤀⬵⤁⬹⤔⬺⤕⬶⤅⬻⤖⬷⤐⬼⤗⬽⤘⤝⤞⤟⤠⤡⤢⤣⤤⤥⤦⤪⤨⤧⤩⤭⤮⤯⤰⤱⤲⤫⤬⬐⬎⬑⬏⤶⤷⥂⥃⥄⭀⥱⥶⥸⭂⭈⭊⥵⭁⭇⭉⥲⭋⭌⥳⥴⥆⥅⥹⥻⬰⥈⬾⥇⬲⟴⥷⭃⥺⭄⥉⥰⬿⤳⥊⥋⥌⥍⥎⥏⥐⥑⥒⥓⥔⥕⥖⥗⥘⥙⥚⥛⥜⥝⥞⥟⥠⥡⥢⥤⥣⥥⥦⥨⥧⥩⥮⥯⥪⥬⥫⥭⤌⤍⤎⤏⬸⤑⬱⟸⟹⟺⤂⤃⤄⤆⤇⤊⤋⭅⭆⟰⟱⇐⇒⇔⇶⟵⟶⟷⬄⬀⬁⬂⬃⬅⬆⬇⬈⬉⬊⬋⬌⬍⟻⟼⤒⤓⤙⤚⤛⤜⥼⥽⥾⥿⤼⤽⤾⤿⤸⤺⤹⤻⥀⥁⟲⟳'
    
    # Weather related symbols
    
    '°℃℉ϟ☀☁☂☃☉☼☽☾♁♨❄❅❆☇☈☄㎎㎏㎜㎝㎞㎡㏄㏎㏑㏒㏕'
    
    # Others
    
    '⋮⋱ↀↁↂ✿✽✻✰✩✦✧♕♕ᐛ♚ᴥᴗᗜㄜ꒪'

    # Emoji Skin Tone

    '🏻🏼🏽🏾🏿'

)

SPECIAL_CHARS = ''.join(sorted(set(SPECIAL_CHARS)))

class CanTokenizer(BaseTokenizer):
    """ Uses Bert WordPiece Tokenizer """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        sep_token: Union[str, AddedToken] = "</s>",
        cls_token: Union[str, AddedToken] = "<s>",
        nl_token: Union[str, AddedToken] = "<nl>",
        pad_token: Union[str, AddedToken] = "<pad>",
        mask_token: Union[str, AddedToken] = "<mask>",
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        separate_numbers: bool = True,
        strip_accents: bool = True,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        special_chars: str = SPECIAL_CHARS,
        zh_norm: bool = True,
    ):

        if vocab_file is not None:
            tokenizer = Tokenizer(WordPiece(vocab_file, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordPiece())

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(nl_token)) is not None:
            tokenizer.add_special_tokens([str(nl_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        tokenizer.normalizer = Sequence([NFKC(), BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            separate_numbers=separate_numbers,
            strip_accents=strip_accents,
            lowercase=lowercase,
            special_chars=special_chars,
            zh_norm=zh_norm
        )])
        tokenizer.pre_tokenizer = BertPreTokenizer()

        
        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "BertWordPiece",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "nl_token": nl_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "separate_numbers": separate_numbers,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "special_chars": special_chars,
            "zh_norm": zh_norm,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 20,
        limit_alphabet: int = 6000,
        initial_alphabet: List[str] = [],
        special_tokens: List[Union[str, AddedToken]] = [
            "<pad>",
            "<unk>",
            "<s>",
            "<nl>",
            "</s>",
            "<mask>",
        ],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given files """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)

        


class CanTokenizerSP(BaseTokenizer):
    """ SentencePiece BPE Tokenizer
    Represents the BPE algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        replacement: str = "▁",
        add_prefix_space: bool = True,
        no_consecutive_space: bool = True,
        dropout: Optional[float] = None,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        separate_numbers: bool = True,
        strip_accents: bool = True,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        special_chars: str = SPECIAL_CHARS,
        zh_norm: bool = True,
    ):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                BPE(vocab_file, merges_file, dropout=dropout, unk_token=unk_token)
            )
        else:
            tokenizer = Tokenizer(BPE())

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        tokenizer.normalizer = Sequence([NFKC(), BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            separate_numbers=separate_numbers,
            strip_accents=strip_accents,
            lowercase=lowercase,
            special_chars=special_chars,
            zh_norm=zh_norm
        )])
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space, no_consecutive_space=no_consecutive_space
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space, no_consecutive_space=no_consecutive_space
        )

        parameters = {
            "model": "SentencePieceBPE",
            "unk_token": unk_token,
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
            "no_consecutive_space": no_consecutive_space,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 20,
        special_tokens: List[Union[str, AddedToken]] = [
            "<pad>",
            "<unk>",
            "<s>",
            "<nl>",
            "</s>",
            "<mask>",
        ],
        limit_alphabet: int = 6000,
        initial_alphabet: List[str] = [],
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)