# Error Analysis — embedding_minilm_gold_llm_label_fresh_holdout

308 false positives and 32 false negatives out of 772 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `neutral_allergen_mention` | 162 | 53% | 180 |
| `illness_mentioned_not_caused_here` | 59 | 19% | 172 |
| `unexplained_fp` | 42 | 14% | 219 |
| `generic_complaint_no_hazard` | 20 | 6% | 139 |
| `negated_hazard` | 15 | 5% | 202 |
| `secondhand_or_hearsay` | 5 | 2% | 247 |
| `hyperbole_or_slang` | 3 | 1% | 258 |
| `strong_negative_sentiment_only` | 1 | 0% | 60 |
| `unpleasant_not_unsafe` | 1 | 0% | 233 |

### `neutral_allergen_mention` — 162 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- Have never been so unimpressed and bothered with a brunch spot. It's like they came to Chicago went to handlebar and thought "this isn't hipster enough we must make a copy of this that's not as good, but ten times as hipster" Once we finally sat down it took 30 minutes for someon… (p=0.999)
- The food is good, definitely the best exclusively gluten free cafe in the area. However, the staff that works the front register is super rude- if I'm spending $6 on a cupcake, you could at least be polite. You'd think at a specialty cafe like this they would have better people w… (p=0.922)
- First off, let me say that I'm neither vegan nor gluten-free. I like to try vegan foods, though, because when there's a reasonable vegan option, I'll generally go for that. Unfortunately, places like Sweet Freedom are the reason people think vegan baking can't be good. I've baked… (p=0.793)

### `illness_mentioned_not_caused_here` — 59 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- Ignorant and inconsiderate!!! Do not even attempt to eat here if you have any type of food allergy or dietary restriction. They are completely inconsiderate of making any accommodations. my friend took her daughter here who is lactose intolerant. She brought her own almond milk f… (p=1.000)
- Ice cream was awsome but more importantly my daughter has severe food allergies. Staff were knowledgeable and had a good process preventing cross contamination.… (p=0.999)
- Wowza. Someone doesn't often expect any South Philly establishment off the beaten path (aka Passyunk) to serve non-food-poisoning-containing raw fish (SPTR aside). But Hibachi2Go breaks the mold and serves decently fresh sashimi, delicious rolls and a slew of tasty salads/teriyak… (p=0.980)

### `unexplained_fp` — 42 cases

No rule matched — requires manual review.

- Went for the first time for lunch, paid $8 for the buffet. Clean large and open environment. Better than alot of the cramped dark Chinese buffets that are much more common and usually end in food poisoning. The food quality was above average for a buffet and everything I saw at t… (p=0.327)
- Cannot believe this hype monster breathes fire with five star reviews. Maybe if you love salty feces slithering down your throat and surly employees who have to "make nice" when the management is present. Christ. Typical cafeteria routine. Think Lubys on a smaller scale. They're … (p=0.687)
- This little brewery ain't so bad! This brewery is located in a pretty industrial area and seems rather huge. My husband and I sat at the bar and was greeted by a very friendly bartender. I ordered the Tucson Blonde which was nothing to write home about. In fact, it was a little f… (p=0.388)

### `generic_complaint_no_hazard` — 20 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- 3.5/5 stars. I really want this place to do well. The man who owns this place is incredibly friendly and makes all the ice cream at the shop. We went during the summer where you would expect ice cream to be relatively higher in demand, but because business was slow, this shop was… (p=0.816)
- When you're on a time crunch, the last thing you need is a slow moving line in a fast food drive-through. As we inched through this particular Steak 'n Shake, we all immediately regretted our decision on choosing this place. Due to being squeezed between two other vehicles, we we… (p=0.650)
- Nice clean store with great selection of organics. People at the register are always friendly too and love that their bakery super careful about cross contamination when it comes to peanuts...at least so they say when I expressed the concern.… (p=0.986)

### `negated_hazard` — 15 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Food is over priced for what you get and server gave me the wrong sandwich! Good thing I'm not allergic to the turkey I didn't want The bread was toasted so much my nePhew cut his mouth! Wait in line and for food is too long! Not worth it at all!!!!!!… (p=0.997)
- One of my go-to's for classic Italian food done right! Some of my favorite dishes include the calamari Sicilian, eggplant tower, grilled romaine salad, eggplant or chicken parm, penne vodka, pumpkin ravioli, sweet potato gnocchis. Their crab cake is my favorite -- served over sau… (p=1.000)
- After reading reviews that Green Basil happily accommodates vegan and vegetarian diets, I ordered delivery specifying that everything must be vegan like this: (***** VEGAN PLEASE *****). Normally if a restaurant cannot accommodate this, they have the courtesy of contacting me to … (p=0.996)

### `secondhand_or_hearsay` — 5 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- Bar Louie protects customers from unprotected servers by ignoring older patrons Due to Covid, I chose mid-afternoon for a RARE and relished venture out. Arrival time, 3:11 ish. Outdoors, one other table occupied. Two servers. Awesome. A third server appeared after about eight min… (p=0.999)
- I'm bordering on 3.5 stars. For some reason I thought this would be a little less fratty than it was. I really enjoyed the booze milkshake (I got the rice crispy one). It's made with graeters ice cream so throw sheets to the wind and abandon any dairy allergy you may have. It was… (p=0.836)
- Not sure if the pizza, it did look amazing. I had a pasta dish with shrimp and saffron which I loved, the arugula salad with mushrooms was just okay- lacking something. The grilled squired was very flavorful and tender. The artichoke appetizer had too much vinegar. We didn't care… (p=0.531)

### `hyperbole_or_slang` — 3 cases

Figurative language reusing hazard vocabulary ('to die for', 'killer tacos'). Purely lexical signal with inverted sentiment — the clearest case for contextual embeddings over TF-IDF.

- "Chicken dum biryani? Can I get some smart biryani?" Vomit-inducing, cringe-inducing, jokes aside, that was one of the highlights of my visit to the lunch buffet at Hyderabad House. Today they had several kinds of Biryani (I wish I could remember the names of the others). The kin… (p=0.990)
- We had an amazing dinner last Friday, in part and thanks to our amazing server in the bar, Brittney. Can't say we expected much despite the 20minute wait and thought it would be like every other "chain" but it was anything but. I have a severe Paprika allergy and eating out can b… (p=0.992)
- Is there any good way to wait for your food at Remedy? They just shout it out when the order is ready, but you don't know if it's actually your food or someone else's who for the same thing. Plus if you're sitting upstairs it's almost impossible to hear anything that's going on d… (p=1.000)

### `strong_negative_sentiment_only` — 1 cases

Highly negative review with no hazard content. The model is partly reading sentiment as hazard, unsurprising given the label used a star-rating gate.

- If I could give this 0 stars it would be more appropriate! Having Celiac is always a struggle to eat out, but I've always managed to find something to eat at every restaurant. Today I was told I could not eat anything in this restaurant!! Really??? Are you just lazy??? I should b… (p=0.969)

### `unpleasant_not_unsafe` — 1 cases

Describes filth or disgust without an adverse event. The boundary is a genuine definitional question — arguably these deserve flagging in a real deployment.

- The Hubby and I did a Delmar Loop date night last night - dinner, movie and desserts. We started the evening at Nico for dinner. I was very curious about this place after reading the reviews on Yelp...and I gotta say, I just don't get it. From a location/ atmosphere/ people watch… (p=0.720)

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 17 | 53% | 111 |
| `positive_review_with_hazard` | 5 | 16% | 136 |
| `negation_misread` | 4 | 12% | 350 |
| `buried_in_long_review` | 3 | 9% | 352 |
| `unsafe_handling_no_illness` | 2 | 6% | 216 |
| `too_short_weak_signal` | 1 | 3% | 25 |

### `unexplained_fn` — 17 cases

No rule matched — requires manual review.

- Chase appears nice on the outside, but the food was mediocre at best, certainly not worth the price. My mom got terrible food poisoning from the ravioli that kept her up all night. The vodka pappardelle and eggplant parm were good, but swimming in sauce. If you are considering It… (p=0.005)
- I was really excited when this place opened! The hours made it impossible for me to get a cupcake from here. When I finally found a day where I could go, I was disappointed. My expectations were very high and the cupcakes just didn't quite meet them. The cupcakes were very dense … (p=0.002)
- I used my Yelp app to find a restaurant open at midnight that was not a bar. Brixx was an intriguing option because of the wood fire, and I like to support places that work to meet dietary needs like vegan or gluten free. The woman who answered was very welcoming to us when I cal… (p=0.036)

### `positive_review_with_hazard` — 5 cases

4-5 star review reporting a hazard. The star gate in the labelling rule means the training data barely contains these, so the model associates hazards with low ratings.

- This is my favorite restaurant of all time!!! The ambiance, the staff, the location and the flavors! Honestly, the best! Except that time the shrimp gave me food poisoning, I forgive them. I just don't order the shrimp. Try the veggie tostada.… (p=0.072)
- Casey makes this place one of my favorite restaurants. Despite me having an allergic reaction to the truffle fries (seriously?) the first time I dined here, Casey was super awesome, off the wall, and friendly. This second visit was no different, Casey was witty, and had us chuckl… (p=0.002)
- Your review helps others learn about great local businesses. Was in the house the last time the Eagles won the NFC championship and it was raucous. loved the crab fries, would only be better if they were house cute. Also had the crab and it was above average. Got there around 1pm… (p=0.071)

### `negation_misread` — 4 cases

Negation cue present but the review is genuinely a hazard ('not the first time I got sick here'). Over-application of the negation pattern.

- The icing tasted great, the cake however was a little too dry - this day, the cupcakes were either over baked or too old or not stored properly? I bought a dozen for my office for around $32. We all had the same complaint on the cake being too dry in this batch. I tried this plac… (p=0.120)
- I'm not sure what I expected before walking in which left me open to what was in store. I expect 25% experience/environment, 25% service, 25% Food Quality and 25% cost/value. It's located downtown, and has a full bar in the front, with room to have appetizers and loiter until get… (p=0.025)
- You know, it's really too bad that I have to give the Hookah House a three-start review. The place is beautiful, and I've often said that it seems like a place that James Bond would have visited. If Bond were to have visited the Hookah House, he'd have enjoyed a seat in one of th… (p=0.004)

### `buried_in_long_review` — 3 cases

Hazard mentioned late in a long review, or diluted by surrounding content. NOTE: the tempting 'past the 256-token window' reading was MEASURED AND REFUTED for the gold false negatives — with the real DebertaV2TokenizerFast at max_length=256, only 1 of 23 residual FNs has its hazard cue past the window, only 2 of 23 exceed 256 tokens at all, and the median cue position is token 39. Raising max_length would recover at most one FN. See results/gold_fn_handread.md. Treat this bucket as dilution/salience, not truncation.

- Much fanfare had been given on the Internet about the burgers at Good Dog, a bar located on 15th street between Walnut and Locust, especially in regards to their Good Dog burger which comes stuffed with Roquefort cheese. I opted for their traditional burger, prepared medium - whi… (p=0.129)
- 10 years ago when I was 20, I walked in with another female companion with a group of 6 or 8 guys (I don't remember) . It was a Bachelor Party and they prettty much dragged us innocent girls in. I'm pretty sure we weren't served although I don't remember (this is becoming the the… (p=0.013)
- Food: 5/5 Service: 2.5-3/5 Overall Score: 4/5 A date and I came here the Friday before Thanksgiving around 7pm for dinner. The place was not full, however the restaurant seemed short-staffed to accommodate the amount of guests. There were long waits to be seated (despite only hal… (p=0.029)

### `unsafe_handling_no_illness` — 2 cases

Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so the label misses a share of these — the model inherits the blind spot.

- I'm giving it two stars just because their staff was awesome and the bloody mary's were good. Other than that, their bacon has literally made my husband and I, never want bacon again. Do NOT order the Bacon Sampler. Like, how do you mess up bacon? All of the flavors tasted like r… (p=0.089)
- I was served a disgusting, half-eaten quarter pounder sandwich from this McDonald's drive through at around 11PM on 28 May 2016. I only found out after I got home and opened the box. I immediately sent an email (with a picture of the food, and the receipt) to the area supervisor'… (p=0.069)

### `too_short_weak_signal` — 1 cases

Very short review; little evidence either way.

- Started a new job in the area and wanted some Chinese. Picked up some lunch from here and Almost threw up afterwards. Will not return!… (p=0.013)
