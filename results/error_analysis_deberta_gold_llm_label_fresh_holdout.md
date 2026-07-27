# Error Analysis — deberta_gold_llm_label_fresh_holdout

197 false positives and 39 false negatives out of 772 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `illness_mentioned_not_caused_here` | 66 | 34% | 164 |
| `neutral_allergen_mention` | 63 | 32% | 203 |
| `generic_complaint_no_hazard` | 23 | 12% | 136 |
| `unexplained_fp` | 23 | 12% | 232 |
| `negated_hazard` | 13 | 7% | 214 |
| `secondhand_or_hearsay` | 4 | 2% | 298 |
| `hyperbole_or_slang` | 3 | 2% | 258 |
| `strong_negative_sentiment_only` | 1 | 0% | 60 |
| `unpleasant_not_unsafe` | 1 | 0% | 233 |

### `illness_mentioned_not_caused_here` — 66 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- Ignorant and inconsiderate!!! Do not even attempt to eat here if you have any type of food allergy or dietary restriction. They are completely inconsiderate of making any accommodations. my friend took her daughter here who is lactose intolerant. She brought her own almond milk f… (p=0.996)
- Ice cream was awsome but more importantly my daughter has severe food allergies. Staff were knowledgeable and had a good process preventing cross contamination.… (p=0.995)
- Wowza. Someone doesn't often expect any South Philly establishment off the beaten path (aka Passyunk) to serve non-food-poisoning-containing raw fish (SPTR aside). But Hibachi2Go breaks the mold and serves decently fresh sashimi, delicious rolls and a slew of tasty salads/teriyak… (p=0.995)

### `neutral_allergen_mention` — 63 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- Have never been so unimpressed and bothered with a brunch spot. It's like they came to Chicago went to handlebar and thought "this isn't hipster enough we must make a copy of this that's not as good, but ten times as hipster" Once we finally sat down it took 30 minutes for someon… (p=0.995)
- Hoe excited was i to find a gluten free vegan pizza (up charge) option. Staff was super accommodating and courteous. When ordering gluten free it is common to be asked Allergy or Preferance. But when i get down the line to add my toppings they actually switched gloves to prevent … (p=0.995)
- Today we went to Semenza's because my mom remembers this as being awesome when she worked for the school around the corner. They seemed to have good deals and it is a cute pizzeria style business. But that is about where it ends sorry to say. We went inside and waited. And waited… (p=0.995)

### `generic_complaint_no_hazard` — 23 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- 3.5/5 stars. I really want this place to do well. The man who owns this place is incredibly friendly and makes all the ice cream at the shop. We went during the summer where you would expect ice cream to be relatively higher in demand, but because business was slow, this shop was… (p=0.990)
- When you're on a time crunch, the last thing you need is a slow moving line in a fast food drive-through. As we inched through this particular Steak 'n Shake, we all immediately regretted our decision on choosing this place. Due to being squeezed between two other vehicles, we we… (p=0.395)
- Nice clean store with great selection of organics. People at the register are always friendly too and love that their bakery super careful about cross contamination when it comes to peanuts...at least so they say when I expressed the concern.… (p=0.989)

### `unexplained_fp` — 23 cases

No rule matched — requires manual review.

- Went for the first time for lunch, paid $8 for the buffet. Clean large and open environment. Better than alot of the cramped dark Chinese buffets that are much more common and usually end in food poisoning. The food quality was above average for a buffet and everything I saw at t… (p=0.995)
- Finding a good restaurant with vegan options in a new town can be difficult. Siam Elephant was intriguing and within walking distance from our hotel, so we decided to give it a try. Siam Elephant is a Thai restaurant, which raised some concerns for me as someone with severe peanu… (p=0.995)
- Disability Accomodations at the Royal Senesta Hotel, New Orleans. June 2011 Disabilities come in all shapes and sizes. Most people think in terms of the well known and visible conditions, such as mobility issues, deafness, blindness... But there are many more little known conditi… (p=0.259)

### `negated_hazard` — 13 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Food is over priced for what you get and server gave me the wrong sandwich! Good thing I'm not allergic to the turkey I didn't want The bread was toasted so much my nePhew cut his mouth! Wait in line and for food is too long! Not worth it at all!!!!!!… (p=0.995)
- One of my go-to's for classic Italian food done right! Some of my favorite dishes include the calamari Sicilian, eggplant tower, grilled romaine salad, eggplant or chicken parm, penne vodka, pumpkin ravioli, sweet potato gnocchis. Their crab cake is my favorite -- served over sau… (p=0.995)
- Creamy, yummy, goodness! After a hot day at the beach and relaxing in the pool, we stopped in for a cool treat at Uncle Andy's Ice Cream Parlor. The ice cream was delightful, portion sizes were generous and overall delicious with a capital D! We ordered a total of 6 different fla… (p=0.995)

### `secondhand_or_hearsay` — 4 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- Bar Louie protects customers from unprotected servers by ignoring older patrons Due to Covid, I chose mid-afternoon for a RARE and relished venture out. Arrival time, 3:11 ish. Outdoors, one other table occupied. Two servers. Awesome. A third server appeared after about eight min… (p=0.798)
- A full celebration tea for 2 contains: 2 pots of tea, a fruit plate,salad and soup, 2 tier tea stand and cupcake with icecream as dessert. The environment is very well decorated , and apparently it's someone's home. When I made a reservation they told me there's only 12 and 1 o'c… (p=0.992)
- I'm bordering on 3.5 stars. For some reason I thought this would be a little less fratty than it was. I really enjoyed the booze milkshake (I got the rice crispy one). It's made with graeters ice cream so throw sheets to the wind and abandon any dairy allergy you may have. It was… (p=0.995)

### `hyperbole_or_slang` — 3 cases

Figurative language reusing hazard vocabulary ('to die for', 'killer tacos'). Purely lexical signal with inverted sentiment — the clearest case for contextual embeddings over TF-IDF.

- "Chicken dum biryani? Can I get some smart biryani?" Vomit-inducing, cringe-inducing, jokes aside, that was one of the highlights of my visit to the lunch buffet at Hyderabad House. Today they had several kinds of Biryani (I wish I could remember the names of the others). The kin… (p=0.208)
- We had an amazing dinner last Friday, in part and thanks to our amazing server in the bar, Brittney. Can't say we expected much despite the 20minute wait and thought it would be like every other "chain" but it was anything but. I have a severe Paprika allergy and eating out can b… (p=0.995)
- Is there any good way to wait for your food at Remedy? They just shout it out when the order is ready, but you don't know if it's actually your food or someone else's who for the same thing. Plus if you're sitting upstairs it's almost impossible to hear anything that's going on d… (p=0.988)

### `strong_negative_sentiment_only` — 1 cases

Highly negative review with no hazard content. The model is partly reading sentiment as hazard, unsurprising given the label used a star-rating gate.

- If I could give this 0 stars it would be more appropriate! Having Celiac is always a struggle to eat out, but I've always managed to find something to eat at every restaurant. Today I was told I could not eat anything in this restaurant!! Really??? Are you just lazy??? I should b… (p=0.993)

### `unpleasant_not_unsafe` — 1 cases

Describes filth or disgust without an adverse event. The boundary is a genuine definitional question — arguably these deserve flagging in a real deployment.

- The Hubby and I did a Delmar Loop date night last night - dinner, movie and desserts. We started the evening at Nico for dinner. I was very curious about this place after reading the reviews on Yelp...and I gotta say, I just don't get it. From a location/ atmosphere/ people watch… (p=0.995)

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 23 | 59% | 106 |
| `buried_in_long_review` | 7 | 18% | 361 |
| `unsafe_handling_no_illness` | 5 | 13% | 161 |
| `mild_understated_wording` | 1 | 3% | 282 |
| `negation_misread` | 1 | 3% | 767 |
| `positive_review_with_hazard` | 1 | 3% | 98 |
| `too_short_weak_signal` | 1 | 3% | 25 |

### `unexplained_fn` — 23 cases

No rule matched — requires manual review.

- On 2/20/16, I encountered this place that my friends raved about. Once we were seated the waitress brought water to the table OMG the PLASTIC cup was cloudy, Grease was floating on top of my water made me want to VOMIT. I asked for a to go cup of water she said those cups are not… (p=0.003)
- I was really excited when this place opened! The hours made it impossible for me to get a cupcake from here. When I finally found a day where I could go, I was disappointed. My expectations were very high and the cupcakes just didn't quite meet them. The cupcakes were very dense … (p=0.001)
- I used my Yelp app to find a restaurant open at midnight that was not a bar. Brixx was an intriguing option because of the wood fire, and I like to support places that work to meet dietary needs like vegan or gluten free. The woman who answered was very welcoming to us when I cal… (p=0.021)

### `buried_in_long_review` — 7 cases

Hazard mentioned late in a long review, past the 256-token truncation window or diluted by surrounding content. Directly actionable: raise max_length.

- i don't think i could live without their breakfast burrito (the barking dog) It is so good. I'm not a big fan of sweet stuff in the morning but their french toast with bananas or pancakes with fruit stuff OR even whatever they are having as a special is always good. I have also l… (p=0.001)
- I took my family to Native Gill & Wings and I have to admit I was not impressed. It was a Wednesday night and maybe 6 other tables of 3/4 top. It took our waitress about 10 mins after we sat down to even come take our drink order, then only brought out 2 of the 5 drinks (water & … (p=0.114)
- Came here for breakfast after a night of drinking with my friends following a wedding. Walked in and it seemed everyone knew each other by first name. The customers and workers seemed to all be on a first name basis which was cool. It has that crappy diner feel which I love. If d… (p=0.012)

### `unsafe_handling_no_illness` — 5 cases

Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so the label misses a share of these — the model inherits the blind spot.

- This review is strictly about the food: it pains me to do this but it needs to be known. I'll start with the positives; The french fries were perfect as well as the cheese and charcuterie plate. The Mac and cheese was bland even with sausage and tasted of only cream (I tried some… (p=0.061)
- I'm giving it two stars just because their staff was awesome and the bloody mary's were good. Other than that, their bacon has literally made my husband and I, never want bacon again. Do NOT order the Bacon Sampler. Like, how do you mess up bacon? All of the flavors tasted like r… (p=0.002)
- Terribly smug attitudes by the cashiers, always. One creamer available, which is usually whole milk in a never clean station. Yucky doesn't stop at the milk stand. Their gimmicky coffee is of a fine chalky rust with a splash of cross-contaminated feces. Harsh but it must be warne… (p=0.199)

### `mild_understated_wording` — 1 cases

Understated symptoms ('didn't sit right with me'). Same ceiling as above, milder.

- This cafe is "line up and order" style. I met 3 people there after an afternoon film at the Garneau. They have several types of chai on the menu but I chose a pomegranate juice. Once you order they give you an electronic buzzer to let you know when your order is ready- I like thi… (p=0.065)

### `negation_misread` — 1 cases

Negation cue present but the review is genuinely a hazard ('not the first time I got sick here'). Over-application of the negation pattern.

- I'm not sure what I expected before walking in which left me open to what was in store. I expect 25% experience/environment, 25% service, 25% Food Quality and 25% cost/value. It's located downtown, and has a full bar in the front, with room to have appetizers and loiter until get… (p=0.019)

### `positive_review_with_hazard` — 1 cases

4-5 star review reporting a hazard. The star gate in the labelling rule means the training data barely contains these, so the model associates hazards with low ratings.

- Your review helps others learn about great local businesses. Was in the house the last time the Eagles won the NFC championship and it was raucous. loved the crab fries, would only be better if they were house cute. Also had the crab and it was above average. Got there around 1pm… (p=0.001)

### `too_short_weak_signal` — 1 cases

Very short review; little evidence either way.

- Started a new job in the area and wanted some Chinese. Picked up some lunch from here and Almost threw up afterwards. Will not return!… (p=0.001)
