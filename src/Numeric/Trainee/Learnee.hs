{-# LANGUAGE RankNTypes, FlexibleContexts, TypeFamilies, MultiParamTypeClasses, FlexibleInstances #-}

module Numeric.Trainee.Learnee (
	CombineResult, Combine(..),
	eval,
	(⇉), (‖), into, paired,
	cost, squared,
	learnee, computee,

	trainOnce, trainBatch, trainUntil
	) where

import Prelude.Unicode

import Control.Arrow ((***))
import Control.Lens
import Data.List (unfoldr)
import Data.Proxy
import Data.Random (runRVar, StdRandom(..), MonadRandom)
import Data.Random.List (shuffle)
import Numeric.AD (AD)
import Numeric.AD.Mode.Forward (Forward, auto, diff')

import Numeric.Trainee.Types

type family CombineResult a b where
	CombineResult () () = ()
	CombineResult a () = a
	CombineResult () b = b
	CombineResult a b = (a, b)

class Combine a b where
	combine ∷ a → b → CombineResult a b
	uncombine ∷ Proxy (a, b) → CombineResult a b → (a, b)

instance Combine () () where
	combine _ _ = ()
	uncombine _ () = ((), ())

instance Combine a () where
	combine x _ = x
	uncombine _ x = (x, ())

instance Combine () b where
	combine _ y = y
	uncombine _ y = ((), y)

instance {-# OVERLAPPABLE #-} (CombineResult a b ~ (a, b)) ⇒ Combine a b where
	combine x y = (x, y)
	uncombine _ (x, y) = (x, y)

eval ∷ Learnee w a b → a → b
eval (Learnee ws f) = fst ∘ f ws

(⇉) ∷ Combine w w' ⇒ Learnee w a b → Learnee w' b c → Learnee (CombineResult w w') a c
Learnee lws f ⇉ Learnee rws g = Learnee ws h where
	h ws' x = (z, up) where
		(lws', rws') = uncombine (proxyOf (lws, rws)) ws'
		(y, f') = f lws' x
		(z, g') = g rws' y
		up dz = (dx, combine lws'' rws'') where
			(dy, rws'') = g' dz
			(dx, lws'') = f' dy
	ws = combine lws rws

into ∷ Combine w w' ⇒ Learnee w a b → Learnee w' b c → Learnee (CombineResult w w') a c
into = (⇉)

(‖) ∷ Combine w w' ⇒ Learnee w a b → Learnee w' a' b' → Learnee (CombineResult w w') (a, a') (b, b')
Learnee lws f ‖ Learnee rws g = Learnee ws h where
	h ws' (x, y) = ((x', y'), up) where
		(lws', rws') = uncombine (proxyOf (lws, rws)) ws'
		(x', f') = f lws' x
		(y', g') = g rws' y
		up (dx', dy') = ((dx, dy), combine lws'' rws'') where
			(dx, lws'') = f' dx'
			(dy, rws'') = g' dy'
	ws = combine lws rws

paired ∷ Combine w w' ⇒ Learnee w a b → Learnee w' a' b' → Learnee (CombineResult w w') (a, a') (b, b')
paired = (‖)

cost ∷ Num a ⇒ (forall s . AD s (Forward a) → AD s (Forward a) → AD s (Forward a)) → Cost a
cost fn x = diff' (fn (auto x))

squared ∷ Num a ⇒ Cost a
squared = cost $ \x y → (x - y) ^ (2 ∷ Integer)

learnee ∷ Gradee (a, w) b → w → Learnee w a b
learnee g ws = Learnee ws h where
	h ws' x = (y, back) where
		y = view (runGradee g) (x, ws')
		back dy = set (runGradee g) dy (x, ws')

computee ∷ Gradee a b → Computee a b
computee g = Learnee () h where
	h _ x = (y, back) where
		y = view (runGradee g) x
		back dy = (set (runGradee g) dy x, ())

learnPass ∷ Learnee w a b → Cost b → a → b → (w, b)
learnPass (Learnee ws f) c x y' = (dws, e) where
	(y, back) = f ws x
	(e, de) = c y' y
	(_, dws) = back de

trainOnce ∷ Params w ⇒ Learnee w a b → Cost b → a → b → (Learnee w a b, b)
trainOnce l c x y' = (over params (plusP dw) l, e) where
	(dw, e) = learnPass l c x y'

trainBatch ∷ (Params w, Fractional b) ⇒ Learnee w a b → Cost b → [(a, b)] → (Learnee w a b, b)
trainBatch l c xs = (over params (plusP dw) l, e) where
	(dw, e) = (sumP *** avg) ∘ unzip ∘ map (uncurry (learnPass l c)) $ xs
	avg ls = sum ls / fromIntegral (length ls)

trainEpoch ∷ (Params w, Fractional b) ⇒ Learnee w a b → Cost b → [[(a, b)]] → (Learnee w a b, b)
trainEpoch l c = foldr (\xs (l', _) → trainBatch l' c xs) (l, 0)

trainUntil ∷ (MonadRandom m, Params w, Fractional b, Ord b) ⇒ Int → Int → b → Learnee w a b → Cost b → [(a, b)] → m (Learnee w a b)
trainUntil 0 _ _ l _ _ = return l
trainUntil epochs batch eps l c xs
	| e < eps = return l'
	| otherwise = do
		xs' ← runRVar (shuffle xs) StdRandom
		trainUntil (pred epochs) batch eps l' c xs'
	where
		(l', e) = trainEpoch l c bs
		bs = unfoldr (Just ∘ splitAt batch) xs




proxyOf ∷ a → Proxy a
proxyOf _ = Proxy
