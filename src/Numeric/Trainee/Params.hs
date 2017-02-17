{-# LANGUAGE GADTs, RankNTypes, ConstraintKinds, TypeOperators, FlexibleInstances #-}

module Numeric.Trainee.Params (
	Parametric,
	Params(..),
	onParams, liftParams,
	onParams2, liftParams2,
	castParams
	) where

import Prelude.Unicode

import Control.DeepSeq
import Data.Function (fix)
import Data.List (intercalate, intersperse)
import Data.Maybe (fromMaybe)
import Data.Ratio ((%))
import Data.Typeable
import qualified Data.Vector as V

-- | Constraints to smth, that can be parameters
type Parametric w = (Show w, Num w, Fractional w, NFData w, Typeable w)

-- | Parameters holder, we use it in @Learnee@ not to pass params type, as long as
-- combining many @Learnee@s will produce huge and unreadable params type
data Params where
	Params ∷ Parametric w ⇒ w → Params
	-- | Implicitly converts to any params in `onParams2`, used as target for `fromIntegral` and `fromRational` implementation
	AnyParam ∷ Rational → Params
	deriving (Typeable)

instance Show Params where
	show (Params ws) = show ws
	show (AnyParam r) = "any[" ++ show r ++ "]"

instance Num Params where
	(+) = liftParams2 (+)
	(*) = liftParams2 (*)
	abs = liftParams abs
	signum = liftParams signum
	fromInteger = AnyParam ∘ flip (%) 1
	negate = liftParams negate

instance Fractional Params where
	fromRational = AnyParam
	recip = liftParams recip

instance NFData Params where
	rnf (Params ws) = rnf ws
	rnf (AnyParam r) = rnf r

instance {-# OVERLAPPING #-} Show (Params, Params) where
	show (l, r) = intercalate "\n" $ intersperse (replicate 10 '-') $
		filter (not ∘ null) [show l, show r]

instance Num (Params, Params) where
	(l, r) + (l', r') = (l + l', r + r')
	(l, r) * (l', r') = (l * l', r * r')
	abs (l, r) = (abs l, abs r)
	signum (l, r) = (signum l, signum r)
	fromInteger i = (fromInteger i, fromInteger i)
	negate (l, r) = (negate l, negate r)

instance Fractional (Params, Params) where
	fromRational r = (fromRational r, fromRational r)
	recip (l, r) = (recip l, recip r)

instance {-# OVERLAPPING #-} Show (V.Vector Params) where
	show ps = intercalate "\n" $ intersperse (replicate 10 '-') $
		filter (not ∘ null) ∘ map show ∘ V.toList $ ps

instance Num (V.Vector Params) where
	ls + rs = uncurry (V.zipWith (+)) $ unifyVecs ls rs
	ls * rs = uncurry (V.zipWith (*)) $ unifyVecs ls rs
	abs = V.map abs
	signum = V.map signum
	fromInteger = V.singleton ∘ fromInteger
	negate = V.map negate

unifyVecs ∷ V.Vector Params → V.Vector Params → (V.Vector Params, V.Vector Params)
unifyVecs x y
	| V.length x ≡ V.length y = (x, y)
	| anyParamVec x ∨ anyParamVec y = (extend x, extend y)
	| otherwise = error $ concat [
		"unifyVecs: params vectors length mismatch: ",
		show $ V.length x,
		" and ",
		show $ V.length y]
	where
		anyParamVec v = V.length v ≡ 1 ∧ case V.head v of
			AnyParam _ → True
			_ → False
		l = max (V.length x) (V.length y)
		extend v
			| anyParamVec v = V.replicate l (V.head v)
			| otherwise = v

instance Fractional (V.Vector Params) where
	fromRational = V.singleton ∘ fromRational
	recip = V.map recip

onParams ∷ (forall w . Parametric w ⇒ w → a) → Params → a
onParams fn (AnyParam r) = fn r
onParams fn (Params ws) = fn ws

liftParams ∷ (forall w . Parametric w ⇒ w → w) → Params → Params
liftParams fn (AnyParam r) = AnyParam $ fn r
liftParams fn p = onParams (Params ∘ fn) p

onParams2 ∷ (forall w . Parametric w ⇒ w → w → a) → Params → Params → a
onParams2 fn (AnyParam lr) (AnyParam rr) = fn lr rr
onParams2 fn (AnyParam lr) (Params rws) = fn (fromRational lr) rws
onParams2 fn (Params lws) (AnyParam rr) = fn lws (fromRational rr)
onParams2 fn (Params lws) (Params rws) = case eqT' lws rws of
	Just Refl → fn lws rws
	_ → error $ "params type mismatch: '" ++ typeName lws ++ "' and '" ++ typeName rws ++ "'"
	where
		eqT' ∷ (Typeable u, Typeable v) ⇒ u → v → Maybe (u :~: v)
		eqT' _ _ = eqT

liftParams2 ∷ (forall w . Parametric w ⇒ w → w → w) → Params → Params → Params
liftParams2 fn (AnyParam lr) (AnyParam rr) = AnyParam $ fn lr rr
liftParams2 fn l r = onParams2 ((Params ∘) ∘ fn) l r

-- | Force cast to specified type
castParams ∷ Parametric a ⇒ Params → a
castParams (AnyParam r) = fromRational r
castParams (Params p) = fromMaybe (castError p) ∘ cast $ p where
	castError ∷ (Parametric u, Parametric v) ⇒ u → v
	castError x = fix $ \y → error $ "castParams: type mismatch, expected '" ++ typeName y ++ "', got '" ++ typeName x ++ "'"

-- Utils

typeName ∷ Typeable a ⇒ a → String
typeName = show ∘ typeRep ∘ proxy'

proxy' ∷ a → Proxy a
proxy' _ = Proxy
